@@directive("'use client'")

@module("tailwind-merge")
external cn: (string, string, string, option<string>) => string = "twMerge"

module Variant = {
  @unboxed
  type t =
    | @as("default") Default
    | @as("secondary") Secondary
    | @as("destructive") Destructive
    | @as("outline") Outline
    | @as("ghost") Ghost
    | @as("link") Link
}

module Size = {
  @unboxed
  type t =
    | @as("default") Default
    | @as("xs") Xs
    | @as("sm") Sm
    | @as("lg") Lg
    | @as("icon") Icon
    | @as("icon-xs") IconXs
    | @as("icon-sm") IconSm
    | @as("icon-lg") IconLg
}

let buttonVariantClass = (~variant: Variant.t) =>
  switch variant {
  | Default => "cn-button-variant-default"
  | Outline => "cn-button-variant-outline"
  | Secondary => "cn-button-variant-secondary"
  | Ghost => "cn-button-variant-ghost"
  | Destructive => "cn-button-variant-destructive"
  | Link => "cn-button-variant-link"
  }

let buttonSizeClass = (~size: Size.t) =>
  switch size {
  | Xs => "cn-button-size-xs"
  | Sm => "cn-button-size-sm"
  | Lg => "cn-button-size-lg"
  | Icon => "cn-button-size-icon"
  | IconXs => "cn-button-size-icon-xs"
  | IconSm => "cn-button-size-icon-sm"
  | IconLg => "cn-button-size-icon-lg"
  | Default => "cn-button-size-default"
  }

let baseClass = "cn-button group/button inline-flex shrink-0 items-center justify-center whitespace-nowrap transition-all outline-none select-none disabled:pointer-events-none disabled:opacity-50 [&_svg]:pointer-events-none [&_svg]:shrink-0"

let buttonVariants = (~variant=Variant.Default, ~size=Size.Default, ~className=?) =>
  cn(baseClass, buttonVariantClass(~variant), buttonSizeClass(~size), className)

type props = {
  variant?: Variant.t,
  size?: Size.t,
  ...BaseUi.Types.BaseUIComponentProps.t,
  ...BaseUi.Types.NativeButtonProps.t,
  focusableWhenDisabled?: bool,
}

let toBaseUiProps: props => BaseUi.Button.props = %raw(`({variant, size, ...rest}) => rest`)

@react.componentWithProps(props)
let make = (props: props) => {
  let variant = props.variant->Option.getOr(Default)
  let size = props.size->Option.getOr(Default)
  let className = props.className
  let baseUiProps = props->toBaseUiProps
  <BaseUi.Button
    {...baseUiProps}
    dataSlot={props.dataSlot->Option.getOr("button")}
    className={buttonVariants(~variant, ~size, ~className?)}
  />
}
