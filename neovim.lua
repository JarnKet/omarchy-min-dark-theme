return {
  {
    "datsfilipe/min-theme.nvim",
    lazy = false,
    priority = 1000,
    opts = {
      theme = "dark",
      -- Min Dark in VS Code sets no italics.
      italics = {
        comments = false,
        keywords = false,
        functions = false,
        strings = false,
        variables = false,
      },
    },
  },
  {
    "LazyVim/LazyVim",
    opts = {
      colorscheme = "min-theme",
    },
  },
}
