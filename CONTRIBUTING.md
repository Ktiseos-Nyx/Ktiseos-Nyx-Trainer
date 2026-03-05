# Contributing Guide

Thank you for your interest in contributing to Ktiseos-Nyx-Trainer. This guide outlines our development process and contribution standards.

## Project Architecture

This is a **web-based LoRA training environment** built with:
- **Frontend**: Next.js 15 (React 19, TypeScript, Tailwind CSS v4, shadcn/ui)
- **Backend API**: FastAPI (Python, Pydantic)
- **Training Backend**: Vendored Kohya SS (`trainer/derrian_backend/`)
- **Deployment**: Local development (Windows/Linux) and VastAI GPU instances

## Ways to Contribute

### Bug Reports
Help us identify and resolve issues in the project.

**Submission Requirements:**
- Search existing issues before creating new reports
- Use provided issue templates when available
- Include environment details: GPU model, OS, Python version, Node.js version, browser
- Specify whether the issue occurs locally, on VastAI, or both
- Provide clear reproduction steps and expected vs actual behavior
- Attach relevant logs, screenshots, or error traces

### Feature Requests
Propose improvements and new functionality.

**Submission Guidelines:**
- Review existing issues and discussions for similar requests
- Clearly articulate the problem your feature solves
- Provide specific implementation details rather than vague descriptions
- Consider feature scope and project compatibility

### Code Contributions
Contribute directly to the codebase through pull requests.

**Priority Areas:**
- **Frontend UI**: React component improvements and user experience enhancements
- **API Layer**: FastAPI endpoint fixes, validation improvements, and new endpoints
- **Dataset Management**: Upload workflows, tagging, and captioning improvements
- **Training Integration**: Kohya SS parameter support, TOML generation fixes
- **Performance**: Memory optimization and training speed improvements
- **Documentation**: Code documentation, user guides, and examples
- **Platform Support**: Cross-platform compatibility (Windows dev, Linux/VastAI deploy)

## Development Process

### Getting Started
1. **Fork and Clone**: Create a personal fork and clone locally
2. **Branch Creation**: Use descriptive branch names (e.g., `feature/dataset-validation`, `fix/training-toml-bug`)
3. **Backend Setup**: Run `python installer.py` to set up the vendored training backend
4. **Frontend Setup**: `cd frontend && npm install`
5. **Start Services**: Run `./start_services_local.sh` (or start FastAPI + Next.js dev servers separately)
6. **Development**: Implement changes following project conventions
7. **Testing**: Verify functionality — `npm run build` for frontend, manual testing for API
8. **Pull Request**: Submit for review with clear description

### Tech Stack Reference

| Layer | Technology | Key Files |
|-------|-----------|-----------|
| Frontend | Next.js 15, React 19, TypeScript | `frontend/app/`, `frontend/components/` |
| UI Components | shadcn/ui (Radix primitives) | `frontend/components/ui/` |
| API Client | Typed fetch wrapper | `frontend/lib/api.ts` |
| Validation | Zod (frontend), Pydantic (backend) | `frontend/lib/validation.ts`, `services/models/` |
| API Routes | FastAPI | `api/routes/` |
| Services | Python business logic | `services/` |
| Training | Vendored Kohya SS + LyCORIS | `trainer/derrian_backend/` |
| Config Generation | TOML | `services/trainers/kohya_toml.py` |

### Code Standards

**TypeScript / Frontend Guidelines:**
- Always use shadcn/ui components — never raw HTML `<input>`, `<select>`, `<button>`, etc.
- Use `@/` path aliases for imports (e.g., `@/components/ui/button`)
- All components must be TypeScript (no plain JavaScript)
- Use `'use client'` directive for components with state, effects, or event handlers
- Follow accessibility best practices (semantic HTML, ARIA attributes, keyboard navigation)
- API calls go through centralized modules in `lib/api.ts`

**Python / Backend Guidelines:**
- Follow PEP 8 coding standards with reasonable flexibility
- Use Pydantic models with `Field()` descriptions for all request/response types
- Use `Literal` types for enum validation on API route models
- Include type hints for function parameters and return values
- Use cross-platform file path operations (`os.path.join()`, `pathlib.Path`)
- Do not lint or reformat vendored code in `trainer/derrian_backend/`

**Three-Layer Architecture:**
When adding or modifying features, ensure consistency across all three layers:
1. **Frontend** (`frontend/lib/api.ts`, `frontend/lib/validation.ts`) — TypeScript types and Zod schemas
2. **API Route** (`api/routes/`) — FastAPI endpoint with route-level Pydantic model
3. **Service Layer** (`services/`) — Business logic with service-level Pydantic models

Field names, types, and defaults must match across all layers.

**Training Integration Requirements:**
- Validate all user inputs before processing
- Consider memory constraints across different hardware configurations
- Ensure cross-platform compatibility (Windows development, Linux/VastAI deployment)
- Maintain backwards compatibility with existing configurations
- Test TOML generation output against Kohya SS script expectations

## Testing Requirements

### Pre-Submission Testing
- **TypeScript Check**: Run `npx tsc --noEmit` in `frontend/` — must pass cleanly
- **Production Build**: Run `npm run build` in `frontend/` — must complete without errors
- **Functionality Verification**: Confirm intended behavior works correctly in the browser
- **Edge Case Testing**: Validate handling of boundary conditions and invalid inputs
- **Regression Testing**: Ensure existing functionality remains unaffected

### Testing Procedures
- Backend: Run `python installer.py` for fresh environment setup
- Frontend: Run `npm install && npm run build` for clean build verification
- Complete workflow testing: setup → dataset preparation → training configuration → training start
- Small dataset testing for rapid iteration
- Error condition testing with invalid or missing inputs

## Communication Standards

### Issue Discussions
- Maintain respectful and constructive communication
- Stay focused on technical topics relevant to the issue
- Provide context about use cases and requirements
- Be patient with response times from volunteer maintainers

### Pull Request Process
- Provide clear explanations of changes and their purpose
- Reference related issues using `Fixes #number` or `Related to #number`
- Respond constructively to code review feedback
- Update documentation to reflect behavioral changes
- Ensure all three layers (frontend, API, service) are updated together for API changes

## Contribution Guidelines

### Acceptable Contributions
- Bug fixes and performance improvements
- Feature implementations discussed in issues
- Documentation improvements and examples
- Test coverage enhancements
- Platform compatibility improvements
- Accessibility improvements

### Unacceptable Contributions
- Malicious or harmful code
- Copyright violations or unauthorized code usage
- Breaking changes without prior discussion and approval
- Spam, self-promotion, or off-topic content
- Duplicate work without coordination
- Modifications to vendored code (`trainer/derrian_backend/`) without discussion

## Recognition

Contributors receive:
- Attribution in project documentation and release notes
- Community recognition for significant contributions
- Maintainer consideration for sustained, high-quality contributions

## Support and Questions

### Getting Help
- **General Questions**: Use GitHub Discussions for project-related inquiries
- **Technical Issues**: Search existing issues before creating new ones
- **Specific Problems**: Comment on relevant pull requests or issues
- **Clarifications**: Request specific information about implementation details

### Response Expectations
- Acknowledge receipt of contributions within reasonable timeframes
- Provide constructive feedback on code quality and project fit
- Maintain transparent communication about acceptance criteria
- Offer guidance for contribution improvements when needed

## Legal and Licensing

### Contribution Agreement
By submitting contributions, you agree that:
- Your contributions are licensed under the project's MIT license
- You have legal rights to submit the contributed code
- Your contributions do not violate third-party copyrights or licenses
- You understand the open-source nature of the project

### Intellectual Property
- Ensure all contributed code is original or properly licensed
- Document any third-party dependencies or libraries
- Respect existing license terms and attribution requirements

---

**Project Philosophy**: We strive to balance accessibility for new users with powerful functionality for advanced practitioners. Contributions should support this goal while maintaining code quality and project stability. This project is built with neurodivergent-friendly workflows in mind — clear structure, consistent patterns, and helpful error messages are valued.

**Contact**: For questions about contributing, please use GitHub Discussions or open an issue for project-specific topics.
