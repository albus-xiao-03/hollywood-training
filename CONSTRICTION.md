# Contrictions

This the constrictions that Claude **MUST** follow while working on the project.

---

## GitHub
The `GitHub` has been fully configured in this project, and ready to work with `Claude`.
    - `Claude` **MUST NOT** change the `user.name` and `user.email` in the project.
    - `Claude` **MUST NOT** use `push` without permission.
    - `Claude` may use `add` and `commit` to manage the files in the project. 
    - `Claude` may use `branch` to create, merge, or delete a branch.
    - `Claude` may change `.gitignore` when necessary.

---

## Temporary File
**ALL** the temporary files, including the environment test, helper functions, and other necessary temporary, **MUST** be stored in the folder `.temp` **INSIDE** the project folder, **NOT** the global environment.

---