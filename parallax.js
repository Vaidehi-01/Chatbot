const parallaxArray = [
  {
    title: "deserunt dolores molestiae 1",
    description:
      "Lorem ipsum dolor sit amet, consectetur adipisicing elit. Obcaecati beatae perferendis deserunt dolores molestiae hic dolorum maxime numquam, ad rerum quas optio quis ut",
    image:
      "https://images.unsplash.com/photo-1547856194-dd2e1bd5d25e?auto=format&fit=crop&q=80&w=1932&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D",
  },
  {
    title: "deserunt dolores molestiae 1",
    description:
      "Lorem ipsum dolor sit amet, consectetur adipisicing elit. Obcaecati beatae perferendis deserunt dolores molestiae hic dolorum maxime numquam, ad rerum quas optio quis ut",
    image:
      "https://images.unsplash.com/photo-1577985051167-0d49eec21977?auto=format&fit=crop&q=80&w=2089&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D",
  },
  {
    title: "deserunt dolores molestiae 1",
    description:
      " Lorem ipsum dolor sit amet, consectetur adipisicing elit. Obcaecati beatae perferendis deserunt dolores molestiae hic dolorum maxime numquam, ad rerum quas optio quis ut",
    image:
      "https://images.unsplash.com/photo-1481627834876-b7833e8f5570?auto=format&fit=crop&q=80&w=2128&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D",
  },
  {
    title: "deserunt dolores molestiae 1",
    description:
      " Lorem ipsum dolor sit amet, consectetur adipisicing elit. Obcaecati beatae perferendis deserunt dolores molestiae hic dolorum maxime numquam, ad rerum quas optio quis ut",
    image:
      "https://images.unsplash.com/photo-1533327325824-76bc4e62d560?auto=format&fit=crop&q=80&w=2069&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D",
  },
  {
    title: "deserunt dolores molestiae 1",
    description:
      "Lorem ipsum dolor sit amet, consectetur adipisicing elit. Obcaecati beatae perferendis deserunt dolores molestiae hic dolorum maxime numquam, ad rerum quas optio quis ut",
    image:
      "https://images.unsplash.com/photo-1414124488080-0188dcbb8834?auto=format&fit=crop&q=80&w=2070&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D",
  },
];
parallaxArray.map(({ description, image, title }) => {
  document.writeln(
    `<div class='parent' style='background:#ffffff url(${image}) no-repeat center center fixed;background-size: cover;>`
  );
  document.writeln("<div style='display:flex; flex-direction:column;'>");
  document.writeln("<h1>" + title + "</h1>");
  document.writeln("<p>" + description + "</p>");
  document.writeln("</div>");
  document.writeln("</div>");
});
