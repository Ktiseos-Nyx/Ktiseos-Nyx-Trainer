"use client"

import Link from "next/link"
import { useTheme } from "next-themes"
import { Zap } from "lucide-react"
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
        {/* Logo - Left */}
        <Link href="/" className="flex items-center gap-2 font-bold text-lg flex-shrink-0">
          <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-gradient-to-br from-purple-600 to-blue-600 shadow-lg">
            <Zap className="h-5 w-5 text-white" />
          </div>
          <span className="bg-gradient-to-r from-purple-600 to-blue-600 bg-clip-text text-transparent whitespace-nowrap">
            KNX TRAINER
          </span>
        </Link>

        {/* Navigation Links - Center (absolute positioning) */}
        <NavigationMenu className="hidden md:flex absolute left-1/2 -translate-x-1/2">
          <NavigationMenuList>
            {/* Dashboard - First! */}
            <NavigationMenuItem>
              <NavigationMenuLink asChild>
                <Link href="/dashboard" className={navigationMenuTriggerStyle()}>
                  Dashboard
                </Link>
              </NavigationMenuLink>
            </NavigationMenuItem>

            {/* Training Tools - Dropdown */}
            <NavigationMenuItem>
              <NavigationMenuTrigger>Training Tools</NavigationMenuTrigger>
              <NavigationMenuContent>
                <ul className="grid w-[400px] gap-3 p-4 md:w-[500px] md:grid-cols-2 lg:w-[600px]">
                  <ListItem href="/dataset" title="Dataset">
                    Upload and prepare training datasets
                  </ListItem>
                  <ListItem href="/training" title="Training">
                    Configure and monitor LoRA training
                  </ListItem>
                  <ListItem href="/calculator" title="Calculator">
                    Calculate optimal training steps
                  </ListItem>
                  <ListItem href="/configs" title="Configs">
                    Load and save training configurations
                  </ListItem>
                </ul>
              </NavigationMenuContent>
            </NavigationMenuItem>

            {/* Utilities - Dropdown */}
            <NavigationMenuItem>
              <NavigationMenuTrigger>Utilities</NavigationMenuTrigger>
              <NavigationMenuContent>
                <ul className="grid w-[400px] gap-3 p-4 md:w-[500px] md:grid-cols-2">
                  <ListItem href="/files" title="File Manager">
                    Browse and manage training files
                  </ListItem>
                  <ListItem href="/utilities" title="LoRA Tools">
                    Resize and upload trained LoRAs
                  </ListItem>
                </ul>
              </NavigationMenuContent>
            </NavigationMenuItem>

            {/* Settings - Top level */}
            <NavigationMenuItem>
              <NavigationMenuLink asChild>
                <Link href="/settings" className={navigationMenuTriggerStyle()}>
                  Settings
                </Link>
              </NavigationMenuLink>
            </NavigationMenuItem>
          </NavigationMenuList>
        </NavigationMenu>

        {/* Theme Switcher - Right */}
        <div className="flex items-center flex-shrink-0">
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
  ...props
}: {
  className?: string
  title: string
  children: React.ReactNode
  href: string
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
          <div className="text-sm font-medium leading-none">{title}</div>
          <p className="line-clamp-2 text-sm leading-snug text-muted-foreground">
            {children}
          </p>
        </Link>
      </NavigationMenuLink>
    </li>
  )
}
