"use client"

import Link from "next/link"
import { useTheme } from "next-themes"
import {
  Zap,
  LayoutDashboard,
  Database,
  TrendingUp,
  Wrench,
  FolderTree,
  BookOpen,
  Info,
  Settings as SettingsIcon,
  FolderOpen,
  Edit,
  Tags,
  Calculator as CalculatorIcon,
  Package,
  HardDrive,
  Download,
  Files
} from "lucide-react"
import {
  NavigationMenu,
  NavigationMenuContent,
  NavigationMenuItem,
  NavigationMenuLink,
  NavigationMenuList,
  NavigationMenuTrigger,
  navigationMenuTriggerStyle,
} from "@/components/ui/navigation-menu"
import { ThemeSwitcher } from "@/components/ui/shadcn-io/theme-switcher"
import { cn } from "@/lib/utils"

export function Navbar() {
  const { theme, setTheme } = useTheme()

  return (
    <nav className="sticky top-0 z-50 w-full border-b border-border/40 bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60">
      <div className="container mx-auto flex h-16 items-center justify-between px-4">
        {/* Left spacer for alignment */}
        <div className="flex-1" />

        {/* Navigation Links - Center */}
        <div className="flex justify-center">
          <NavigationMenu>
            <NavigationMenuList>
            {/* Dashboard - First! */}
            <NavigationMenuItem>
              <NavigationMenuLink asChild>
                <Link href="/dashboard" className={navigationMenuTriggerStyle()}>
                  <LayoutDashboard className="w-4 h-4 mr-2" />
                  Dashboard
                </Link>
              </NavigationMenuLink>
            </NavigationMenuItem>

             {/* Dataset Tools - Dropdown */}
            <NavigationMenuItem>
              <NavigationMenuTrigger>
                <Database className="w-4 h-4 mr-2" />
                Dataset Tools
              </NavigationMenuTrigger>
              <NavigationMenuContent>
                <ul className="grid w-[400px] gap-3 p-4 md:w-[500px] md:grid-cols-2 lg:w-[600px]">
                  <ListItem href="/dataset" title="Dataset" icon={<FolderOpen className="w-4 h-4" />}>
                    Upload and prepare training datasets
                  </ListItem>
                  <ListItem href="/dataset/auto-tag" title="Auto-Tag" icon={<Zap className="w-4 h-4" />}>
                    Auto-generate tags using WD14 models
                  </ListItem>
                  <ListItem href="/dataset/tags" title="Tag Editor" icon={<Tags className="w-4 h-4" />}>
                    Manage image tags and captions
                  </ListItem>
                </ul>
              </NavigationMenuContent>
            </NavigationMenuItem>

            {/* Training Tools - Dropdown */}
            <NavigationMenuItem>
              <NavigationMenuTrigger>
                <TrendingUp className="w-4 h-4 mr-2" />
                Training Tools
              </NavigationMenuTrigger>
              <NavigationMenuContent>
                <ul className="grid w-[400px] gap-3 p-4 md:w-[500px] md:grid-cols-2 lg:w-[600px]">
                  <ListItem href="/training" title="Training" icon={<Zap className="w-4 h-4" />}>
                    Configure and monitor LoRA training
                  </ListItem>
                  <ListItem href="/calculator" title="Calculator" icon={<CalculatorIcon className="w-4 h-4" />}>
                    Calculate optimal training steps
                  </ListItem>
                </ul>
              </NavigationMenuContent>
            </NavigationMenuItem>

            {/* Utilities - Dropdown */}
            <NavigationMenuItem>
              <NavigationMenuTrigger>
                <Wrench className="w-4 h-4 mr-2" />
                Utilities
              </NavigationMenuTrigger>
              <NavigationMenuContent>
                <ul className="grid w-[400px] gap-3 p-4 md:w-[500px] md:grid-cols-2">
                  <ListItem href="/utilities" title="LoRA Tools" icon={<Package className="w-4 h-4" />}>
                    Resize and upload trained LoRAs
                  </ListItem>
                  <ListItem href="/models" title="Models" icon={<HardDrive className="w-4 h-4" />}>
                    Manage downloaded models and VAEs
                  </ListItem>
                  <ListItem href="/models/browse" title="Civitai Downloader" icon={<Download className="w-4 h-4" />}>
                    Download models from Civitai
                  </ListItem>
                </ul>
              </NavigationMenuContent>
            </NavigationMenuItem>

            {/* File Manager - Top level */}
            <NavigationMenuItem>
            <NavigationMenuTrigger>
              <FolderTree className="w-4 h-4 mr-2" />
              File Management
            </NavigationMenuTrigger>
            <NavigationMenuContent>
              <ul className="grid w-[400px] gap-3 p-4 md:w-[500px] md:grid-cols-1">
                  <ListItem href="/files" title="File Manager" icon={<Files className="w-4 h-4" />}>
                    Use the File Manager to upload, download, and manage files!
                  </ListItem>
              </ul>
              </NavigationMenuContent>
            </NavigationMenuItem>

            {/* Docs - Top level */}
            <NavigationMenuItem>
              <NavigationMenuLink asChild>
                <Link href="/docs" className={navigationMenuTriggerStyle()}>
                  <BookOpen className="w-4 h-4 mr-2" />
                  Docs
                </Link>
              </NavigationMenuLink>
            </NavigationMenuItem>

            {/* About - Top level */}
            <NavigationMenuItem>
              <NavigationMenuLink asChild>
                <Link href="/about" className={navigationMenuTriggerStyle()}>
                  <Info className="w-4 h-4 mr-2" />
                  About
                </Link>
              </NavigationMenuLink>
            </NavigationMenuItem>

            {/* Settings - Top level */}
            <NavigationMenuItem>
              <NavigationMenuLink asChild>
                <Link href="/settings" className={navigationMenuTriggerStyle()}>
                  <SettingsIcon className="w-4 h-4 mr-2" />
                  Settings
                </Link>
              </NavigationMenuLink>
            </NavigationMenuItem>
          </NavigationMenuList>
        </NavigationMenu>
        </div>

        {/* Theme Switcher - Right */}
        <div className="flex items-center justify-end flex-1">
          <ThemeSwitcher
            value={theme as "light" | "dark" | "system"}
            onChange={(newTheme) => setTheme(newTheme)}
          />
        </div>
      </div>
    </nav>
  )
}

const ListItem = ({
  className,
  title,
  children,
  href,
  icon,
  ...props
}: {
  className?: string
  title: string
  children: React.ReactNode
  href: string
  icon?: React.ReactNode
}) => {
  return (
    <li>
      <NavigationMenuLink asChild>
        <Link
          href={href}
          className={cn(
            "block select-none space-y-1 rounded-md p-3 leading-none no-underline outline-none transition-colors hover:bg-accent hover:text-accent-foreground focus:bg-accent focus:text-accent-foreground",
            className
          )}
          {...props}
        >
          <div className="text-sm font-medium leading-none flex items-center gap-2">
            {icon}
            {title}
          </div>
          <p className="line-clamp-2 text-sm leading-snug text-muted-foreground">
            {children}
          </p>
        </Link>
      </NavigationMenuLink>
    </li>
  )
}
