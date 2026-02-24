<template>
  <span
    ref="triggerRef"
    class="inline-flex cursor-pointer items-center gap-1 rounded-full px-3 py-1 text-xs font-medium transition-colors"
    :class="badgeClass"
    @mouseenter="showTooltip"
    @mouseleave="hideTooltip"
    @click="toggleTooltip"
  >
    <component :is="badgeIcon" class="h-3 w-3" />
    {{ badgeText }}
  </span>
  <Teleport to="body">
    <Transition
      enter-active-class="transition-opacity duration-150"
      enter-from-class="opacity-0"
      enter-to-class="opacity-100"
      leave-active-class="transition-opacity duration-100"
      leave-from-class="opacity-100"
      leave-to-class="opacity-0"
    >
      <div
        v-if="visible"
        class="fixed z-[9999] w-64 rounded-lg border border-border bg-card p-3 shadow-lg"
        :style="tooltipStyle"
        @mouseenter="handleTooltipEnter"
        @mouseleave="handleTooltipLeave"
      >
        <!-- 箭头 -->
        <span
          class="absolute left-1/2 top-full h-0 w-0 -translate-x-1/2 border-x-[6px] border-t-[6px] border-x-transparent border-t-border"
        ></span>
        <span
          class="absolute left-1/2 top-full h-0 w-0 -translate-x-1/2 -translate-y-px border-x-[5px] border-t-[5px] border-x-transparent border-t-card"
        ></span>

        <div class="mb-2 text-xs font-medium text-foreground">配额详情</div>
        <div class="mb-2 flex items-center justify-between text-[11px] text-muted-foreground">
          <span>受限 {{ quotaStatus.limited_count }}/{{ quotaStatus.total_count }}</span>
          <span v-if="quotaStatus.is_expired" class="text-red-500">账号已过期/禁用</span>
        </div>
        <div class="space-y-2">
          <div v-for="(status, type) in quotaStatus.quotas" :key="type" class="flex items-center justify-between text-xs">
            <span class="flex items-center gap-1.5">
              <component :is="getQuotaIconComponent(type as string)" class="h-3.5 w-3.5" :class="getQuotaIconClass(type as string)" />
              <span class="text-muted-foreground">{{ getQuotaName(type as string) }}</span>
            </span>
            <span class="flex items-center gap-1" :class="getStatusClass(status)">
              <component :is="getStatusIcon(status)" class="h-3 w-3" />
              {{ getStatusText(status, type as string) }}
            </span>
          </div>
        </div>
      </div>
    </Transition>
  </Teleport>
</template>

<script setup lang="ts">
import { computed, h, nextTick, ref } from 'vue'
import type { AccountQuotaStatus, QuotaStatus } from '@/types/api'

const props = defineProps<{
  quotaStatus: AccountQuotaStatus
}>()

const triggerRef = ref<HTMLElement | null>(null)
const visible = ref(false)
const tooltipStyle = ref<Record<string, string>>({})
let hideTimeout: ReturnType<typeof setTimeout> | null = null

// SVG Icon Components
const CheckCircleIcon = () => h('svg', { viewBox: '0 0 20 20', fill: 'currentColor' }, [
  h('path', { 'fill-rule': 'evenodd', d: 'M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z', 'clip-rule': 'evenodd' })
])

const BanIcon = () => h('svg', { viewBox: '0 0 20 20', fill: 'currentColor' }, [
  h('path', { 'fill-rule': 'evenodd', d: 'M13.477 14.89A6 6 0 015.11 6.524l8.367 8.368zm1.414-1.414L6.524 5.11a6 6 0 018.367 8.367zM18 10a8 8 0 11-16 0 8 8 0 0116 0z', 'clip-rule': 'evenodd' })
])

const ClockIcon = () => h('svg', { viewBox: '0 0 20 20', fill: 'currentColor' }, [
  h('path', { 'fill-rule': 'evenodd', d: 'M10 18a8 8 0 100-16 8 8 0 000 16zm1-12a1 1 0 10-2 0v4a1 1 0 00.293.707l2.828 2.829a1 1 0 101.415-1.415L11 9.586V6z', 'clip-rule': 'evenodd' })
])

const ChatIcon = () => h('svg', { viewBox: '0 0 20 20', fill: 'currentColor' }, [
  h('path', { 'fill-rule': 'evenodd', d: 'M18 10c0 3.866-3.582 7-8 7a8.841 8.841 0 01-4.083-.98L2 17l1.338-3.123C2.493 12.767 2 11.434 2 10c0-3.866 3.582-7 8-7s8 3.134 8 7zM7 9H5v2h2V9zm8 0h-2v2h2V9zM9 9h2v2H9V9z', 'clip-rule': 'evenodd' })
])

const ImageIcon = () => h('svg', { viewBox: '0 0 20 20', fill: 'currentColor' }, [
  h('path', { 'fill-rule': 'evenodd', d: 'M4 3a2 2 0 00-2 2v10a2 2 0 002 2h12a2 2 0 002-2V5a2 2 0 00-2-2H4zm12 12H4l4-8 3 6 2-4 3 6z', 'clip-rule': 'evenodd' })
])

const VideoIcon = () => h('svg', { viewBox: '0 0 20 20', fill: 'currentColor' }, [
  h('path', { d: 'M2 6a2 2 0 012-2h6a2 2 0 012 2v8a2 2 0 01-2 2H4a2 2 0 01-2-2V6zM14.553 7.106A1 1 0 0014 8v4a1 1 0 00.553.894l2 1A1 1 0 0018 13V7a1 1 0 00-1.447-.894l-2 1z' })
])

const updatePosition = () => {
  if (!triggerRef.value) return
  const rect = triggerRef.value.getBoundingClientRect()
  const offset = 8
  tooltipStyle.value = {
    left: `${rect.left + rect.width / 2}px`,
    top: `${rect.top - offset}px`,
    transform: 'translate(-50%, -100%)',
  }
}

const showTooltip = () => {
  if (hideTimeout) {
    clearTimeout(hideTimeout)
    hideTimeout = null
  }
  visible.value = true
  nextTick(() => {
    updatePosition()
  })
}

const hideTooltip = () => {
  hideTimeout = setTimeout(() => {
    visible.value = false
  }, 150)
}

const handleTooltipEnter = () => {
  if (hideTimeout) {
    clearTimeout(hideTimeout)
    hideTimeout = null
  }
}

const handleTooltipLeave = () => {
  visible.value = false
}

const toggleTooltip = () => {
  if (visible.value) {
    visible.value = false
  } else {
    showTooltip()
  }
}

const badgeClass = computed(() => {
  const { limited_count, total_count } = props.quotaStatus
  if (limited_count === 0) return 'bg-green-500/10 text-green-500'
  if (limited_count === total_count) return 'bg-red-500/10 text-red-500'
  return 'bg-amber-500/10 text-amber-500'
})

const badgeIcon = computed(() => {
  const { limited_count, total_count } = props.quotaStatus
  if (limited_count === 0) return CheckCircleIcon
  if (limited_count === total_count) return BanIcon
  return ClockIcon
})

const badgeText = computed(() => {
  const { limited_count, total_count, quotas, is_expired } = props.quotaStatus

  if (limited_count === 0) {
    return '全部可用'
  }

  if (is_expired && limited_count === total_count) {
    return '已过期/禁用'
  }

  if (limited_count === total_count) {
    return '全部冷却'
  }

  const limitedTypes: string[] = []
  if (!quotas.text.available) limitedTypes.push(formatLimitedType('text', quotas.text.remaining_seconds))
  if (!quotas.images.available) limitedTypes.push(formatLimitedType('images', quotas.images.remaining_seconds))
  if (!quotas.videos.available) limitedTypes.push(formatLimitedType('videos', quotas.videos.remaining_seconds))

  return `冷却 ${limitedTypes.join(' / ')}`
})

const getQuotaIconComponent = (type: string) => {
  const icons: Record<string, ReturnType<typeof h>> = {
    text: ChatIcon(),
    images: ImageIcon(),
    videos: VideoIcon()
  }
  return icons[type] || ChatIcon()
}

const getQuotaIconClass = (type: string) => {
  const classes: Record<string, string> = {
    text: 'text-blue-500',
    images: 'text-purple-500',
    videos: 'text-pink-500'
  }
  return classes[type] || 'text-muted-foreground'
}

const getQuotaName = (type: string) => {
  const names: Record<string, string> = { text: '对话', images: '绘图', videos: '视频' }
  return names[type] || type
}

const getStatusClass = (status: QuotaStatus) => {
  if (status.available) {
    return 'text-green-500 font-medium'
  }
  return status.remaining_seconds ? 'text-amber-500 font-medium' : 'text-red-500 font-medium'
}

const getStatusIcon = (status: QuotaStatus) => {
  if (status.available) {
    return CheckCircleIcon
  }
  if (status.remaining_seconds) {
    return ClockIcon
  }
  return BanIcon
}

const getStatusText = (status: QuotaStatus, type?: string) => {
  // 有配额计数信息时统一显示 used/limit
  if (status.daily_limit != null && status.daily_limit > 0) {
    return `${status.daily_used ?? 0}/${status.daily_limit}`
  }

  if (status.available) {
    return '正常'
  }

  if (status.remaining_seconds) {
    return formatTime(status.remaining_seconds)
  }

  return type ? `${getQuotaName(type)}不可用` : '已过期'
}

const formatTime = (seconds: number) => {
  const h = Math.floor(seconds / 3600)
  const m = Math.floor((seconds % 3600) / 60)
  if (h > 0) {
    return `${h}h ${m}m`
  }
  return `${m}m`
}

const formatLimitedType = (type: string, remaining?: number) => {
  const names: Record<string, string> = { text: '对话', images: '绘图', videos: '视频' }
  const name = names[type] || type
  if (remaining) {
    return `${name}${formatTime(remaining)}`
  }
  return name
}
</script>
