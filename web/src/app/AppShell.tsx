import { ArrowDown, ArrowUp, Moon, Palette, Sun, SunMoon } from 'lucide-react'
import { useEffect, useState } from 'react'
import { NavLink, Outlet, useLocation } from 'react-router'

import { Button } from '@/components/ui/button'
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuGroup,
  DropdownMenuLabel,
  DropdownMenuRadioGroup,
  DropdownMenuRadioItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu'
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select'
import { Separator } from '@/components/ui/separator'
import {
  Sidebar,
  SidebarContent,
  SidebarGroup,
  SidebarGroupContent,
  SidebarGroupLabel,
  SidebarHeader,
  SidebarInset,
  SidebarMenu,
  SidebarMenuButton,
  SidebarMenuItem,
  SidebarProvider,
  SidebarRail,
  SidebarTrigger,
  useSidebar,
} from '@/components/ui/sidebar'
import { Tooltip, TooltipContent, TooltipTrigger } from '@/components/ui/tooltip'
import { getPageJumpSlots, type ScrollPositionState } from '@/domain/detailUi'
import { paletteGroups, paletteName } from '@/domain/teamPalettes'
import { cn } from '@/utils/cn'

import { NAV_ITEMS } from './nav'
import { useTheme, type Theme } from './ThemeProvider'
import { useSeason, withSeason } from './useSeason'

const THEME_ICONS = { light: Sun, dark: Moon, system: SunMoon } as const
const THEME_ORDER: Theme[] = ['light', 'dark', 'system']
const SCROLL_EDGE_THRESHOLD = 8

function ThemeToggle() {
  const { theme, setTheme } = useTheme()
  const Icon = THEME_ICONS[theme]
  const next = THEME_ORDER[(THEME_ORDER.indexOf(theme) + 1) % THEME_ORDER.length]
  return (
    <Tooltip>
      <TooltipTrigger asChild>
        <Button
          variant="ghost"
          size="icon"
          aria-label={`Theme: ${theme}. Switch to ${next}`}
          onClick={() => setTheme(next)}
        >
          <Icon className="size-4" />
        </Button>
      </TooltipTrigger>
      <TooltipContent>Theme: {theme}</TooltipContent>
    </Tooltip>
  )
}

/** The palette menu: the default palette, then every team's, grouped by division. */
function PalettePicker() {
  const { palette, setPalette } = useTheme()
  const name = paletteName(palette)
  return (
    <DropdownMenu>
      <Tooltip>
        <TooltipTrigger asChild>
          <DropdownMenuTrigger asChild>
            <Button variant="ghost" size="icon" aria-label={`Palette: ${name}`}>
              <Palette className={cn('size-4', palette !== 'classic' && 'text-primary')} />
            </Button>
          </DropdownMenuTrigger>
        </TooltipTrigger>
        <TooltipContent>Palette: {name}</TooltipContent>
      </Tooltip>
      <DropdownMenuContent align="end" className="max-h-[70vh] w-64 overflow-y-auto">
        <DropdownMenuRadioGroup value={palette} onValueChange={setPalette}>
          <DropdownMenuRadioItem value="classic">Default</DropdownMenuRadioItem>
          {paletteGroups().map((group) => (
            <DropdownMenuGroup key={group.division}>
              <DropdownMenuSeparator />
              <DropdownMenuLabel className="text-xs font-medium text-muted-foreground">{group.division}</DropdownMenuLabel>
              {group.teams.map((team) => (
                <DropdownMenuRadioItem key={team.id} value={team.id}>
                  <span aria-hidden className="flex gap-0.5">
                    {team.colors.map((color) => (
                      <span key={color} className="size-2.5 rounded-full ring-1 ring-border" style={{ background: color }} />
                    ))}
                  </span>
                  {team.name}
                </DropdownMenuRadioItem>
              ))}
            </DropdownMenuGroup>
          ))}
        </DropdownMenuRadioGroup>
      </DropdownMenuContent>
    </DropdownMenu>
  )
}

function SeasonSelect() {
  const { season, seasons, setSeason } = useSeason()
  if (seasons.length === 0) return null
  return (
    <Select value={season === null ? undefined : String(season)} onValueChange={(value) => setSeason(Number(value))}>
      <SelectTrigger size="sm" className="w-28" aria-label="Season">
        <SelectValue placeholder="Season" />
      </SelectTrigger>
      <SelectContent>
        {seasons.map((candidate) => (
          <SelectItem key={candidate} value={String(candidate)}>
            {candidate}
          </SelectItem>
        ))}
      </SelectContent>
    </Select>
  )
}

function readScrollPosition(): ScrollPositionState {
  const maxScrollTop = Math.max(document.documentElement.scrollHeight - window.innerHeight, 0)
  if (maxScrollTop <= SCROLL_EDGE_THRESHOLD) return { atBottom: true, atTop: true, canScroll: false }
  return {
    atBottom: maxScrollTop - window.scrollY <= SCROLL_EDGE_THRESHOLD,
    atTop: window.scrollY <= SCROLL_EDGE_THRESHOLD,
    canScroll: true,
  }
}

/** Floating jump-to-top and jump-to-bottom buttons for the long stat tables. */
function PageJumpButtons() {
  const [position, setPosition] = useState<ScrollPositionState>({
    atBottom: false,
    atTop: true,
    canScroll: false,
  })

  useEffect(() => {
    const update = () => setPosition(readScrollPosition())
    update()
    window.addEventListener('scroll', update, { passive: true })
    window.addEventListener('resize', update)
    const observer = new ResizeObserver(update)
    observer.observe(document.body)
    return () => {
      window.removeEventListener('scroll', update)
      window.removeEventListener('resize', update)
      observer.disconnect()
    }
  }, [])

  const slots = getPageJumpSlots(position)
  if (slots.length === 0) return null
  return (
    <div className="fixed right-4 bottom-4 z-30 flex flex-col gap-2" aria-label="Page navigation shortcuts">
      {slots.map((slot) => (
        <Button
          key={slot.direction}
          variant="secondary"
          size="icon"
          className={cn('shadow-md', !slot.visible && 'invisible')}
          aria-hidden={!slot.visible}
          tabIndex={slot.visible ? 0 : -1}
          aria-label={slot.direction === 'up' ? 'Scroll to top' : 'Scroll to bottom'}
          onClick={() =>
            window.scrollTo({
              top: slot.direction === 'up' ? 0 : document.documentElement.scrollHeight,
              behavior: 'smooth',
            })
          }
        >
          {slot.direction === 'up' ? <ArrowUp /> : <ArrowDown />}
        </Button>
      ))}
    </div>
  )
}

function AppSidebar() {
  const location = useLocation()
  const { season } = useSeason()
  const { setOpenMobile, isMobile } = useSidebar()
  return (
    <Sidebar collapsible="icon" variant="sidebar">
      <SidebarHeader>
        <div className="flex items-center gap-2 px-1 py-1">
          <img src="/favicon.svg" alt="" className="size-8 shrink-0 rounded-lg" />
          <div className="min-w-0 group-data-[collapsible=icon]:hidden">
            <div className="truncate text-sm font-semibold leading-tight">NFL SOS Ratings</div>
            <div className="truncate text-xs text-muted-foreground">Schedule-adjusted ratings</div>
          </div>
        </div>
      </SidebarHeader>
      <SidebarContent>
        <SidebarGroup>
          <SidebarGroupLabel>Navigate</SidebarGroupLabel>
          <SidebarGroupContent>
            <SidebarMenu>
              {NAV_ITEMS.map((item) => (
                <SidebarMenuItem key={item.to}>
                  <SidebarMenuButton
                    asChild
                    isActive={location.pathname.startsWith(item.to)}
                    tooltip={item.label}
                  >
                    <NavLink to={withSeason(item.to, season)} onClick={() => isMobile && setOpenMobile(false)}>
                      <item.icon />
                      <span>{item.label}</span>
                    </NavLink>
                  </SidebarMenuButton>
                </SidebarMenuItem>
              ))}
            </SidebarMenu>
          </SidebarGroupContent>
        </SidebarGroup>
      </SidebarContent>
      <SidebarRail />
    </Sidebar>
  )
}

/** Sidebar + header + routed content. */
export function AppShell() {
  return (
    <SidebarProvider>
      <AppSidebar />
      <SidebarInset className="min-w-0">
        <header className="sticky top-0 z-20 flex h-14 items-center gap-2 border-b bg-background/85 px-3 backdrop-blur sm:px-4">
          <SidebarTrigger aria-label="Toggle navigation" />
          <Separator orientation="vertical" className="mr-1 h-5" />
          <SeasonSelect />
          <div className="ml-auto flex items-center gap-1">
            <PalettePicker />
            <ThemeToggle />
          </div>
        </header>
        <main className="min-w-0 flex-1 px-3 py-4 sm:px-6 sm:py-6 safe-bottom">
          <Outlet />
        </main>
      </SidebarInset>
      <PageJumpButtons />
    </SidebarProvider>
  )
}
