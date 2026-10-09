(() => {
  const prefixPattern = /\[[^\]\n]*\|[^|\]\n]+\|([A-Z])\|[^\]\n]*\]/g;

  function decorateLogOutput(root = document) {
    root.querySelectorAll(".srh-log code").forEach((code) => {
      if (code.dataset.srhLogDecorated === "true") return;

      const walker = document.createTreeWalker(code, NodeFilter.SHOW_TEXT);
      const textNodes = [];
      while (walker.nextNode()) textNodes.push(walker.currentNode);

      textNodes.forEach((node) => {
        const source = node.nodeValue;
        prefixPattern.lastIndex = 0;
        if (!prefixPattern.test(source)) return;

        prefixPattern.lastIndex = 0;
        const fragment = document.createDocumentFragment();
        let cursor = 0;
        for (const match of source.matchAll(prefixPattern)) {
          fragment.append(source.slice(cursor, match.index));
          const prefix = document.createElement("span");
          prefix.className = `srh-log-prefix srh-log-level-${match[1].toLowerCase()}`;
          prefix.textContent = match[0];
          fragment.append(prefix);
          cursor = match.index + match[0].length;
        }
        fragment.append(source.slice(cursor));
        node.replaceWith(fragment);
      });

      code.dataset.srhLogDecorated = "true";
    });
  }

  document.addEventListener("DOMContentLoaded", () => decorateLogOutput());
  if (typeof document$ !== "undefined") document$.subscribe(() => decorateLogOutput());
})();
