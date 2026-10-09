window.MathJax = {
  tex: {
    inlineMath: [["\\(", "\\)"]],
    displayMath: [["\\[", "\\]"], ["$$", "$$"]],
  },
  options: {
    ignoreHtmlClass: "\\bdata-no-mathjax\\b",
    processHtmlClass: "\\bdata-mathjax\\b",
  },
};

document$.subscribe(() => {
  MathJax.typesetPromise();
});
