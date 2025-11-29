# Frontend Development Guidelines

**Objective:** To ensure the Next.js user interface is consistent, accessible, and maintainable. All frontend development should adhere to these principles.

---

## 1. Core Principles

### 1.1. Accessibility First

Accessibility is not an afterthought; it is a core requirement. Our goal is to create an application that is usable by everyone, including people with disabilities who may use screen readers or rely on keyboard navigation.

#### Guidelines:

-   **Semantic HTML:** Use semantic HTML elements (`<main>`, `<nav>`, `<button>`, `<article>`, etc.) correctly. This provides meaning and structure for assistive technologies.
-   **Keyboard Navigation:** All interactive elements (buttons, links, form fields) **must** be fully operable with a keyboard. This includes visible focus states (outlines) so users know where they are on the page.
-   **ARIA Roles:** Use ARIA (Accessible Rich Internet Applications) attributes where appropriate, especially for custom components, to define their roles and states (e.g., `role="alert"`, `aria-busy="true"`).
-   **Image `alt` Text:** All `<img>` tags must have a descriptive `alt` attribute. If an image is purely decorative, use an empty `alt=""`.
-   **Labels for Inputs:** All form inputs (`<input>`, `<textarea>`, `<select>`) must have an associated `<label>` to describe their purpose.
-   **Color Contrast:** Ensure that text has sufficient color contrast against its background to be readable by people with low vision. Use tools to check contrast ratios.

### 1.2. Component Consistency with shadcn/ui

To maintain a consistent visual style and user experience, we will rely heavily on the **shadcn/ui** component library. This ensures that buttons, forms, modals, and other UI elements look and behave predictably across the entire application.

#### The Rule:

**Always use a `shadcn/ui` component if one exists for your use case.** Before building a new component from scratch, consult the project's `frontend/SHADCN_COMPONENTS.md` file and the official [shadcn/ui documentation](https://ui.shadcn.com/docs/components).

#### When to Build Custom Components:

-   Only create a new, custom component when no existing `shadcn/ui` component meets the specific requirement.
-   When creating a custom component, it should be built by composing other `shadcn/ui` primitives if possible, to maintain a consistent look and feel.
-   All new custom components must adhere to the accessibility guidelines outlined above.

By following these principles, we will create a frontend that is not only visually appealing and consistent but also robust, accessible, and easier to maintain in the long run.
