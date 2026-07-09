import * as React from 'react'
import { Slot } from '@radix-ui/react-slot'
import { cva, type VariantProps } from 'class-variance-authority'

import { cn } from '@/lib/utils'

const clusterVariants = cva('gap-gap bg-transparent', {
  variants: {
    display: {
      flex: 'flex',
      'inline-flex': 'inline-flex',
    },
    direction: {
      row: 'flex-row',
      col: 'flex-col',
    },
    align: {
      start: 'items-start',
      center: 'items-center',
      end: 'items-end',
      stretch: 'items-stretch',
      baseline: 'items-baseline',
    },
    wrap: {
      true: 'flex-wrap',
      false: '',
    },
    bg: {
      muted: '[--cluster-bg:var(--color-muted)]',
      accent: '[--cluster-bg:var(--color-accent)]',
    },
  },
  defaultVariants: {
    display: 'flex',
    direction: 'row',
    align: 'center',
    bg: 'muted',
    wrap: false,
  },
})

type ClusterProps = React.ComponentProps<'div'> & {
  asChild?: boolean
} & VariantProps<typeof clusterVariants>

function Cluster({
  className,
  display,
  direction,
  align,
  wrap,
  bg,
  asChild = false,
  ...props
}: ClusterProps) {
  const Comp = asChild ? Slot : 'div'

  return (
    <Comp
      data-slot="cluster"
      className={cn(
        clusterVariants({ display, direction, align, wrap, bg }),
        className
      )}
      {...props}
    />
  )
}

type FillerProps = Omit<React.ComponentProps<'div'>, 'role' | 'aria-hidden'>

function Filler({ className, ...props }: FillerProps) {
  return (
    <div
      data-slot="cluster-filler"
      role="presentation"
      aria-hidden="true"
      className={cn('min-w-0 flex-1 self-stretch', className)}
      {...props}
    />
  )
}

export { Cluster, Filler, clusterVariants }
export type { ClusterProps, FillerProps }
