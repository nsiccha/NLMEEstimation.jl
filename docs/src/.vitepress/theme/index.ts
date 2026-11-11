import DefaultTheme from 'vitepress/theme'
import type { Theme } from 'vitepress'
import 'virtual:mathjax-styles.css'
import './style.css'
import './docstrings.css'

// Keep generated same-origin figures in the reader's selected VitePress theme.
// This small adapter can be replaced by DocsKit's shared figure integration.
function syncFigureThemes() {
  const dark = document.documentElement.classList.contains('dark')
  const background = getComputedStyle(document.documentElement).getPropertyValue('--vp-c-bg')
  document.querySelectorAll<HTMLIFrameElement>('iframe[data-docs-figure]').forEach(frame => {
    const sync = () => {
      const root = frame.contentDocument?.documentElement
      if (root) {
        root.classList.toggle('dark', dark)
        root.style.setProperty('--docs-background', background)
      }
    }
    frame.onload = sync
    sync()
  })
}
export default {
  extends: DefaultTheme,
  enhanceApp({ router }) {
    if (typeof window !== 'undefined') {
      router.onAfterRouteChanged = () => requestAnimationFrame(syncFigureThemes)
      new MutationObserver(syncFigureThemes).observe(document.documentElement, { attributes: true, attributeFilter: ['class'] })
      window.addEventListener('DOMContentLoaded', syncFigureThemes)
    }
  },
} satisfies Theme
