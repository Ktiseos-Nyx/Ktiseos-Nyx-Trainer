@@jsxConfig({version: 4, mode: "automatic", module_: "BaseUi.BaseUiJsxDOM"})

@module("tailwind-merge")
external cn: (string, option<string>) => string = "twMerge"

module State = {
  @unboxed
  type t =
    | @as("idle") Idle
    | @as("uploading") Uploading
    | @as("processing") Processing
    | @as("error") Error
    | @as("done") Done
}

module Size = {
  @unboxed
  type t =
    | @as("default") Default
    | @as("sm") Sm
    | @as("xs") Xs
}

module Orientation = {
  @unboxed
  type t =
    | @as("horizontal") Horizontal
    | @as("vertical") Vertical
}

module MediaVariant = {
  @unboxed
  type t =
    | @as("icon") Icon
    | @as("image") Image
}

let sizeClass = (~size: Size.t) =>
  switch size {
  | Default => "cn-attachment-size-default"
  | Sm => "cn-attachment-size-sm"
  | Xs => "cn-attachment-size-xs"
  }

let orientationClass = (~orientation: Orientation.t) =>
  switch orientation {
  | Horizontal => "cn-attachment-orientation-horizontal items-center"
  | Vertical => "cn-attachment-orientation-vertical flex-col"
  }

@react.component
let make = (
  ~className=?,
  ~state=State.Done,
  ~size=Size.Default,
  ~orientation=Orientation.Horizontal,
  ~children=?,
  ~id=?,
  ~style=?,
  ~onClick=?,
  ~onKeyDown=?,
) =>
  <div
    ?id
    ?style
    ?onClick
    ?onKeyDown
    ?children
    dataSlot="attachment"
    dataState={(state :> string)}
    dataSize={(size :> string)}
    dataOrientation={(orientation :> string)}
    className={cn(
      `cn-attachment group/attachment relative flex max-w-full min-w-0 shrink-0 flex-wrap border bg-card text-card-foreground transition-colors has-[>a,>button]:hover:bg-muted/50 data-[state=error]:border-destructive/30 data-[state=idle]:border-dashed ${sizeClass(
          ~size,
        )} ${orientationClass(~orientation)}`,
      className,
    )}
  />

module Media = {
  let variantClass = (~variant: MediaVariant.t) =>
    switch variant {
    | Icon => "cn-attachment-media-variant-icon"
    | Image => "cn-attachment-media-variant-image *:[img]:aspect-square *:[img]:w-full *:[img]:object-cover"
    }

  @react.component
  let make = (
    ~className=?,
    ~variant=MediaVariant.Icon,
    ~children=?,
    ~id=?,
    ~style=?,
    ~onClick=?,
    ~onKeyDown=?,
  ) =>
    <div
      ?id
      ?style
      ?onClick
      ?onKeyDown
      ?children
      dataSlot="attachment-media"
      dataVariant={(variant :> string)}
      className={cn(
        `cn-attachment-media relative flex aspect-square shrink-0 items-center justify-center overflow-hidden group-data-[state=error]/attachment:bg-destructive/10 group-data-[state=error]/attachment:text-destructive [&_svg]:pointer-events-none ${variantClass(
            ~variant,
          )}`,
        className,
      )}
    />
}

module Content = {
  @react.component
  let make = (~className=?, ~children=?, ~id=?, ~style=?, ~onClick=?, ~onKeyDown=?) =>
    <div
      ?id
      ?style
      ?onClick
      ?onKeyDown
      ?children
      dataSlot="attachment-content"
      className={cn("cn-attachment-content max-w-full min-w-0 flex-1", className)}
    />
}

module Title = {
  @react.component
  let make = (~className=?, ~children=?, ~id=?, ~style=?, ~onClick=?, ~onKeyDown=?) =>
    <span
      ?id
      ?style
      ?onClick
      ?onKeyDown
      ?children
      dataSlot="attachment-title"
      className={cn(
        "cn-attachment-title block max-w-full min-w-0 truncate group-data-[state=processing]/attachment:shimmer group-data-[state=uploading]/attachment:shimmer",
        className,
      )}
    />
}

module Description = {
  @react.component
  let make = (~className=?, ~children=?, ~id=?, ~style=?, ~onClick=?, ~onKeyDown=?) =>
    <span
      ?id
      ?style
      ?onClick
      ?onKeyDown
      ?children
      dataSlot="attachment-description"
      className={cn(
        "cn-attachment-description block max-w-full min-w-0 truncate text-muted-foreground group-data-[state=error]/attachment:text-destructive/80",
        className,
      )}
    />
}

module Actions = {
  @react.component
  let make = (~className=?, ~children=?, ~id=?, ~style=?, ~onClick=?, ~onKeyDown=?) =>
    <div
      ?id
      ?style
      ?onClick
      ?onKeyDown
      ?children
      dataSlot="attachment-actions"
      className={cn("cn-attachment-actions flex shrink-0 items-center", className)}
    />
}

module Action = {
  @react.component
  let make = (~className=?, ~children=?, ~ariaLabel=?, ~id=?, ~style=?, ~onClick=?, ~onKeyDown=?) =>
    <Button
      ?id
      ?style
      ?onClick
      ?onKeyDown
      ?ariaLabel
      dataSlot="attachment-action"
      variant=Ghost
      size=IconXs
      className={cn("cn-attachment-action", className)}
    >
      {children->Option.getOr(React.null)}
    </Button>
}

module Trigger = {
  @react.component
  let make = (
    ~className=?,
    ~children=?,
    ~ariaLabel=?,
    ~id=?,
    ~style=?,
    ~onClick=?,
    ~onKeyDown=?,
    ~type_="button",
  ) =>
    <button
      ?id
      ?style
      ?onClick
      ?onKeyDown
      ?ariaLabel
      type_
      dataSlot="attachment-trigger"
      className={cn("absolute inset-0 z-10 outline-none", className)}
    >
      {children->Option.getOr(React.null)}
    </button>
}

module Group = {
  @react.component
  let make = (~className=?, ~children=?, ~id=?, ~style=?, ~onClick=?, ~onKeyDown=?) =>
    <div
      ?id
      ?style
      ?onClick
      ?onKeyDown
      ?children
      dataSlot="attachment-group"
      className={cn(
        "flex min-w-0 scroll-fade-x snap-x snap-mandatory scrollbar-none overflow-x-auto overscroll-x-contain *:data-[slot=attachment]:flex-none *:data-[slot=attachment]:snap-start",
        className,
      )}
    />
}
