# Git Workflow

本项目通过 GitHub 在 Windows 台式机和 MacBook 之间同步。核心习惯是：

- 换电脑前一定 `push`
- 换电脑后第一件事一定 `pull`
- 如果 `git status` 显示有未提交改动，先不要直接 `pull`，先提交或确认这些改动是否要保留

## 开始工作

每次打开项目、准备开始工作时，先执行：

```bash
git status
git pull origin main
```

## 结束工作

每次结束工作、准备换电脑或关闭项目时，执行：

```bash
git status
git add -A
git commit -m "Commit by haocunDesktop: describe your changes"
git push origin main
```

如果是在 MacBook 上提交，把 commit message 写成：

```bash
git commit -m "Commit by haocunMacBook: describe your changes"
```

## 注意事项

- `describe your changes` 要改成这次实际做了什么，例如 `update training plots` 或 `add validation results`
- 如果没有任何改动，`git commit` 会提示 nothing to commit，这时不需要提交
- `.pth`、缓存文件、系统文件已经被 `.gitignore` 忽略，正常训练结果里的 CSV、PNG、JSON、HTML 仍会同步
