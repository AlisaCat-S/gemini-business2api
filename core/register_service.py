import asyncio
import logging
import os
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional

from core.account import load_accounts_from_source
from core.base_task_service import BaseTask, BaseTaskService, TaskCancelledError, TaskStatus
from core.config import config
from core.mail_providers import create_temp_mail_client
from core.gemini_automation import GeminiAutomation
from core.proxy_utils import parse_proxy_setting

logger = logging.getLogger("gemini.register")

# 配置检查间隔（秒）
CONFIG_CHECK_INTERVAL_SECONDS = 60


@dataclass
class RegisterTask(BaseTask):
    """注册任务数据类"""
    count: int = 0
    domain: Optional[str] = None
    mail_provider: Optional[str] = None

    def to_dict(self) -> dict:
        """转换为字典"""
        base_dict = super().to_dict()
        base_dict["count"] = self.count
        base_dict["domain"] = self.domain
        base_dict["mail_provider"] = self.mail_provider
        return base_dict


class RegisterService(BaseTaskService[RegisterTask]):
    """注册服务类"""

    def __init__(
        self,
        multi_account_mgr,
        http_client,
        user_agent: str,
        retry_policy,
        session_cache_ttl_seconds: int,
        global_stats_provider: Callable[[], dict],
        set_multi_account_mgr: Optional[Callable[[Any], None]] = None,
    ) -> None:
        super().__init__(
            multi_account_mgr,
            http_client,
            user_agent,
            retry_policy,
            session_cache_ttl_seconds,
            global_stats_provider,
            set_multi_account_mgr,
            log_prefix="REGISTER",
        )
        self._is_polling = False

    def _get_running_task(self) -> Optional[RegisterTask]:
        """获取正在运行或等待中的任务"""
        for task in self._tasks.values():
            if isinstance(task, RegisterTask) and task.status in (TaskStatus.PENDING, TaskStatus.RUNNING):
                return task
        return None

    async def start_register(self, count: Optional[int] = None, domain: Optional[str] = None, mail_provider: Optional[str] = None) -> RegisterTask:
        """
        启动注册任务 - 统一任务管理
        - 如果有正在运行的任务，将新数量添加到该任务
        - 如果没有正在运行的任务，创建新任务
        """
        async with self._lock:
            if os.environ.get("ACCOUNTS_CONFIG"):
                raise ValueError("已设置 ACCOUNTS_CONFIG 环境变量，注册功能已禁用")

            # 先确定使用哪个邮箱服务提供商
            mail_provider_value = (mail_provider or "").strip().lower()
            if not mail_provider_value:
                mail_provider_value = (config.basic.temp_mail_provider or "duckmail").lower()

            # 确定使用哪个域名（优先入参，其次使用各提供商配置）
            domain_value = (domain or "").strip()
            if not domain_value:
                _domain_map = {
                    "duckmail": config.basic.register_domain,
                    "gptmail": config.basic.gptmail_domain,
                    "moemail": config.basic.moemail_domain,
                    "freemail": config.basic.freemail_domain,
                }
                domain_value = (_domain_map.get(mail_provider_value) or "").strip() or None

            register_count = count or config.basic.register_default_count
            register_count = max(1, int(register_count))

            # 检查是否有正在运行的任务
            running_task = self._get_running_task()

            if running_task:
                # 将新数量添加到现有任务
                running_task.count += register_count
                self._append_log(
                    running_task,
                    "info",
                    f"📝 添加 {register_count} 个账户到现有任务 (总计: {running_task.count})"
                )
                return running_task

            # 创建新任务
            task = RegisterTask(id=str(uuid.uuid4()), count=register_count, domain=domain_value, mail_provider=mail_provider_value)
            self._tasks[task.id] = task
            self._append_log(task, "info", f"📝 创建注册任务 (数量: {register_count}, 域名: {domain_value or 'default'}, 提供商: {mail_provider_value})")

            # 直接启动任务
            self._current_task_id = task.id
            asyncio.create_task(self._run_task_directly(task))
            return task

    async def _run_task_directly(self, task: RegisterTask) -> None:
        """直接执行任务"""
        try:
            await self._run_one_task(task)
        finally:
            # 任务完成后清理
            async with self._lock:
                if self._current_task_id == task.id:
                    self._current_task_id = None

    def _execute_task(self, task: RegisterTask):
        return self._run_register_async(task, task.domain, task.mail_provider)

    async def _run_register_async(self, task: RegisterTask, domain: Optional[str], mail_provider: Optional[str]) -> None:
        """异步执行注册任务（支持取消）。"""
        loop = asyncio.get_running_loop()
        self._append_log(task, "info", f"🚀 注册任务已启动 (共 {task.count} 个账号)")

        for idx in range(task.count):
            if task.cancel_requested:
                self._append_log(task, "warning", f"register task cancelled: {task.cancel_reason or 'cancelled'}")
                task.status = TaskStatus.CANCELLED
                task.finished_at = time.time()
                return

            try:
                self._append_log(task, "info", f"📊 进度: {idx + 1}/{task.count}")
                result = await loop.run_in_executor(self._executor, self._register_one, domain, mail_provider, task)
            except TaskCancelledError:
                task.status = TaskStatus.CANCELLED
                task.finished_at = time.time()
                return
            except Exception as exc:
                result = {"success": False, "error": str(exc)}
            task.progress += 1
            task.results.append(result)

            if result.get("success"):
                task.success_count += 1
                email = result.get('email', '未知')
                self._append_log(task, "info", f"✅ 注册成功: {email}")
            else:
                task.fail_count += 1
                error = result.get('error', '未知错误')
                self._append_log(task, "error", f"❌ 注册失败: {error}")

        if task.cancel_requested:
            task.status = TaskStatus.CANCELLED
        else:
            task.status = TaskStatus.SUCCESS if task.fail_count == 0 else TaskStatus.FAILED
        task.finished_at = time.time()
        self._current_task_id = None
        self._append_log(task, "info", f"🏁 注册任务完成 (成功: {task.success_count}, 失败: {task.fail_count}, 总计: {task.count})")

    def _register_one(self, domain: Optional[str], mail_provider: Optional[str], task: RegisterTask) -> dict:
        """注册单个账户"""
        log_cb = lambda level, message: self._append_log(task, level, message)

        log_cb("info", "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        log_cb("info", "🆕 开始注册新账户")
        log_cb("info", "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

        # 使用传递的邮件提供商参数，如果未提供则从配置读取
        temp_mail_provider = (mail_provider or "").strip().lower()
        if not temp_mail_provider:
            temp_mail_provider = (config.basic.temp_mail_provider or "duckmail").lower()

        log_cb("info", f"📧 步骤 1/3: 注册临时邮箱 (提供商={temp_mail_provider})...")

        if temp_mail_provider == "freemail" and not config.basic.freemail_jwt_token:
            log_cb("error", "❌ Freemail JWT Token 未配置")
            return {"success": False, "error": "Freemail JWT Token 未配置"}

        client = create_temp_mail_client(
            temp_mail_provider,
            domain=domain,
            log_cb=log_cb,
        )

        if not client.register_account(domain=domain):
            log_cb("error", f"❌ {temp_mail_provider} 邮箱注册失败")
            return {"success": False, "error": f"{temp_mail_provider} 注册失败"}

        log_cb("info", f"✅ 邮箱注册成功: {client.email}")

        headless = config.basic.browser_headless
        proxy_for_auth, _ = parse_proxy_setting(config.basic.proxy_for_auth)

        log_cb("info", f"🌐 步骤 2/3: 启动浏览器 (无头模式={headless})...")

        automation = GeminiAutomation(
            user_agent=self.user_agent,
            proxy=proxy_for_auth,
            headless=headless,
            log_callback=log_cb,
        )
        # 允许外部取消时立刻关闭浏览器
        self._add_cancel_hook(task.id, lambda: getattr(automation, "stop", lambda: None)())

        try:
            log_cb("info", "🔐 步骤 3/3: 执行 Gemini 自动登录...")
            result = automation.login_and_extract(client.email, client)
        except Exception as exc:
            log_cb("error", f"❌ 自动登录异常: {exc}")
            return {"success": False, "error": str(exc)}

        if not result.get("success"):
            error = result.get("error", "自动化流程失败")
            log_cb("error", f"❌ 自动登录失败: {error}")
            return {"success": False, "error": error}

        log_cb("info", "✅ Gemini 登录成功，正在保存配置...")

        config_data = result["config"]
        config_data["mail_provider"] = temp_mail_provider
        config_data["mail_address"] = client.email

        # 保存邮箱自定义配置
        if temp_mail_provider == "freemail":
            config_data["mail_password"] = ""
            config_data["mail_base_url"] = config.basic.freemail_base_url
            config_data["mail_jwt_token"] = config.basic.freemail_jwt_token
            config_data["mail_verify_ssl"] = config.basic.freemail_verify_ssl
            config_data["mail_domain"] = config.basic.freemail_domain
        elif temp_mail_provider == "gptmail":
            config_data["mail_password"] = ""
            config_data["mail_base_url"] = config.basic.gptmail_base_url
            config_data["mail_api_key"] = config.basic.gptmail_api_key
            config_data["mail_verify_ssl"] = config.basic.gptmail_verify_ssl
            config_data["mail_domain"] = config.basic.gptmail_domain
        elif temp_mail_provider == "moemail":
            config_data["mail_password"] = getattr(client, "email_id", "") or getattr(client, "password", "")
            config_data["mail_base_url"] = config.basic.moemail_base_url
            config_data["mail_api_key"] = config.basic.moemail_api_key
            config_data["mail_domain"] = config.basic.moemail_domain
        elif temp_mail_provider == "duckmail":
            config_data["mail_password"] = getattr(client, "password", "")
            config_data["mail_base_url"] = config.basic.duckmail_base_url
            config_data["mail_api_key"] = config.basic.duckmail_api_key
        else:
            config_data["mail_password"] = getattr(client, "password", "")

        accounts_data = load_accounts_from_source()
        updated = False
        for acc in accounts_data:
            if acc.get("id") == config_data["id"]:
                acc.update(config_data)
                updated = True
                break
        if not updated:
            accounts_data.append(config_data)

        self._apply_accounts_update(accounts_data)

        log_cb("info", "✅ 配置已保存到数据库")
        log_cb("info", "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        log_cb("info", f"🎉 账户注册完成: {client.email}")
        log_cb("info", "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

        return {"success": True, "email": client.email, "config": config_data}

    # ==================== 定时注册轮询 ====================

    @staticmethod
    def _parse_start_time(value: str) -> tuple:
        """解析 HH:MM 格式的起始时间，返回 (hour, minute)"""
        try:
            parts = (value or "").strip().split(":")
            if len(parts) != 2:
                raise ValueError("invalid format")
            hour, minute = int(parts[0]), int(parts[1])
            if not (0 <= hour <= 23 and 0 <= minute <= 59):
                raise ValueError("out of range")
            return hour, minute
        except Exception:
            logger.warning("[REGISTER] 无效的起始时间格式: %s，回退到 00:00", value)
            return 0, 0

    @staticmethod
    def _calc_next_trigger(now: datetime, start_time: str, interval_hours: int) -> datetime:
        """计算下一次触发时间：以当天 start_time 为锚点，按 interval_hours 递推"""
        hour, minute = RegisterService._parse_start_time(start_time)
        interval = timedelta(hours=max(1, interval_hours))
        anchor = now.replace(hour=hour, minute=minute, second=0, microsecond=0)

        if now < anchor:
            return anchor

        elapsed = now - anchor
        steps = int(elapsed.total_seconds() // interval.total_seconds()) + 1
        return anchor + interval * steps

    async def start_polling(self) -> None:
        """启动定时注册轮询（常驻后台）"""
        if self._is_polling:
            logger.warning("[REGISTER] 定时注册轮询已在运行")
            return

        self._is_polling = True
        logger.info("[REGISTER] 定时注册轮询已启动")
        try:
            while self._is_polling:
                # 检查配置是否启用
                if not config.retry.scheduled_register_enabled:
                    logger.debug("[REGISTER] 定时注册未启用，跳过检查")
                    await asyncio.sleep(CONFIG_CHECK_INTERVAL_SECONDS)
                    continue

                # ACCOUNTS_CONFIG 存在时跳过
                if os.environ.get("ACCOUNTS_CONFIG"):
                    logger.debug("[REGISTER] ACCOUNTS_CONFIG 已设置，定时注册跳过")
                    await asyncio.sleep(CONFIG_CHECK_INTERVAL_SECONDS)
                    continue

                # 计算下一次触发时间
                now = datetime.now()
                next_trigger = self._calc_next_trigger(
                    now,
                    config.retry.scheduled_register_start_time,
                    config.retry.scheduled_register_interval_hours,
                )
                wait_seconds = (next_trigger - now).total_seconds()
                logger.info(
                    "[REGISTER] 下次定时注册: %s（%.0f秒后）",
                    next_trigger.strftime("%Y-%m-%d %H:%M:%S"),
                    wait_seconds,
                )

                # 等待到触发时间，期间每分钟检查配置变更
                while wait_seconds > 0 and self._is_polling:
                    sleep_time = min(wait_seconds, CONFIG_CHECK_INTERVAL_SECONDS)
                    await asyncio.sleep(sleep_time)
                    wait_seconds -= sleep_time

                    # 配置被关闭则跳出等待
                    if not config.retry.scheduled_register_enabled:
                        logger.info("[REGISTER] 定时注册已被禁用，中断等待")
                        break

                if not self._is_polling or not config.retry.scheduled_register_enabled:
                    continue

                # 先在锁内检查是否有运行中任务，避免与手动注册竞态
                count = config.retry.scheduled_register_count
                provider = config.retry.scheduled_register_mail_provider or None
                sched_domain = (config.retry.scheduled_register_domain or "").strip() or None
                async with self._lock:
                    running = self._get_running_task()
                    if running:
                        logger.info("[REGISTER] 已有注册任务运行中（%s），跳过本轮定时注册", running.id)
                        skip = True
                    else:
                        skip = False
                if skip:
                    continue
                logger.info("[REGISTER] 定时注册触发: 数量=%d, 邮箱供应商=%s, 域名=%s", count, provider or "默认", sched_domain or "default")
                try:
                    await self.start_register(count=count, mail_provider=provider, domain=sched_domain)
                except Exception as exc:
                    logger.error("[REGISTER] 定时注册失败: %s", exc)

        except asyncio.CancelledError:
            logger.info("[REGISTER] 定时注册轮询已停止")
        except Exception as exc:
            logger.error("[REGISTER] 定时注册轮询异常: %s", exc)
        finally:
            self._is_polling = False

    def stop_polling(self) -> None:
        """停止定时注册轮询"""
        self._is_polling = False
        logger.info("[REGISTER] 正在停止定时注册轮询")
