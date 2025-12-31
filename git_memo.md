# mainのマージ方法
1. `git merge main`
2. `git restore --source=HEAD -- ".devcontainer" ".vscode" ".gitignore" "pyproject.toml" "test"`

# (mainでの)devのマージ方法
1. `git checkout dev -- ./ReadMe.Md`