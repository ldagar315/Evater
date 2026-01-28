# Evater Design Guidelines

This document outlines the design language and standards extracted from the Evater codebase (`HomePage.tsx`, `tailwind.config.js`, `index.css`). Use these guidelines to ensure consistency across all pages.

## 1. Color Palette

### Primary Colors (Teal)

Used for primary actions, highlights, and brand identity.

- **Main**: `text-primary-500` / `bg-primary-500` (#37c2a3)
- **Accent**: `text-primary-600` (#0d9488)
- **Backgrounds**: `bg-primary-50` (#f0fdfa)
- **Borders**: `border-primary-200` (#99f6e4)

### Secondary Colors (Yellow)

Used for warnings, secondary highlights, or specific feature accents.

- **Main**: `bg-secondary-500` (#f4dc90)
- **Backgrounds**: `bg-secondary-50` (#fefce8)
- **Text**: `text-secondary-600` (#d97706)
- **Borders**: `border-secondary-200` (#fde68a)

### Neutral & Base Colors

- **Background (App)**: `bg-cream` (#fbfaf3) - _Global app background_
- **Surface (Cards)**: `bg-white` (#ffffff)
- **Text (Dark)**: `text-dark` (#272426) - _Primary text color_
- **Text (Muted)**: `text-neutral-500` (#737373) - _Secondary text/descriptions_
- **Borders**: `border-neutral-100` (#f5f5f5) or `border-neutral-200` (#e5e5e5)

### Other Accents

- **Purple**: `text-purple-600`, `bg-purple-50`, `border-purple-200` (Used for "AI Viva Session", "Previous Tests")
- **Blue**: `text-blue-600`, `bg-blue-50`, `border-blue-200` (Used for "Previous Feedbacks")

## 2. Typography

**Font Family**: `"Plus Jakarta Sans", system-ui, sans-serif` (Applied via `font-sans` class)

### Headings

- **Page Title**: `text-4xl md:text-5xl font-bold text-dark tracking-tight`
- **Section Title**: `text-2xl font-bold text-dark`
- **Card Title**: `text-lg font-bold text-dark`

### Body Text

- **Lead/Subtitle**: `text-xl text-neutral-500 leading-relaxed font-medium`
- **Standard Text**: `text-sm text-neutral-500 leading-relaxed`
- **Labels**: `text-sm font-semibold text-neutral-500 uppercase tracking-wider`

## 3. Layout & Spacing

### Page Structure

- **Wrapper**: `min-h-screen bg-cream font-sans`
- **Container**: `max-w-7xl mx-auto py-12 px-4 sm:px-6 lg:px-8`

### Spacing Standards

- **Section Bottom Margin**: `mb-16`
- **Component Internal Padding**: `p-6` or `p-8` (Cards)
- **Grid Gaps**: `gap-8`

## 4. Components & UI Elements

### Cards (Navigation/Feature Options)

- **Container**:
  ```tsx
  className =
    "group relative bg-white rounded-2xl p-6 border border-neutral-100 shadow-sm transition-all duration-300 hover:shadow-md hover:-translate-y-1 cursor-pointer";
  ```
- **Icon Container**:
  - Size: `w-12 h-12`
  - Shape: `rounded-xl`
  - Style: `flex items-center justify-center`
  - Dynamic Color: `bg-{color}-50` (e.g., `bg-primary-50`)

### Buttons (Primary Action)

- **Style**:
  ```tsx
  className =
    "inline-flex items-center px-8 py-3 bg-dark text-white font-semibold rounded-xl hover:bg-neutral-800 transition-all duration-200 shadow-lg shadow-neutral-200 hover:shadow-xl transform hover:-translate-y-0.5";
  ```

### Stats/Info Cards

- **Container**: `bg-white rounded-2xl shadow-sm border border-neutral-100 p-8`
- **Divider**: `divide-y md:divide-y-0 md:divide-x divide-neutral-100`

## 5. Animations & Transitions

### Global

- All elements have `transition-colors duration-200` by default (from `index.css`).

### Hover Effects

- **Cards**: Lift up (`-translate-y-1`) and increase shadow (`shadow-md`).
- **Buttons**: Lift up slightly (`-translate-y-0.5`) and increase shadow (`shadow-xl`).

### Custom Keyframes (from `index.css`)

- **Fade In**: `animate-fade-in` (0.3s ease-out)
- **Slide Up**: `animate-slide-up` (0.2s ease-out)

## 6. Scrollbar Customization

- **Width**: `8px`
- **Track**: `bg-cream` (#fbfaf3)
- **Thumb**: `bg-primary-500` (#37c2a3) with `rounded-4px`
- **Thumb Hover**: `#2dd4bf`

## 7. Responsiveness

- **Grid Layouts**: `grid-cols-1 md:grid-cols-2 lg:grid-cols-3`
- **Typography**: `text-4xl md:text-5xl`
- **Dividers**: `divide-y md:divide-y-0 md:divide-x` (Stacks vertically on mobile, horizontally on desktop)
