// Shows a `pylfit-stepper` container's figures one at a time, with
// Previous and Next buttons. The figures share one grid cell, so the
// box is as tall as the tallest and the page below never moves.
// Without JavaScript every figure shows, in order.
document.addEventListener("DOMContentLoaded", () => {
  for (const stepper of document.querySelectorAll(".pylfit-stepper")) {
    const frames = [...stepper.querySelectorAll(":scope > figure")];
    if (frames.length < 2) continue;
    const button = (label) => {
      const b = document.createElement("button");
      b.type = "button";
      b.textContent = label;
      return b;
    };
    const previous = button("Previous");
    const next = button("Next");
    const counter = document.createElement("span");
    counter.setAttribute("aria-live", "polite");
    const nav = document.createElement("div");
    nav.className = "pylfit-stepper-nav";
    nav.append(previous, counter, next);
    const stack = document.createElement("div");
    stack.className = "pylfit-stepper-frames";
    stack.append(...frames);
    stepper.append(nav, stack);
    stepper.classList.add("pylfit-stepper-on");
    let current = 0;
    const show = (i) => {
      current = i;
      frames.forEach((frame, j) => {
        frame.classList.toggle("pylfit-stepper-current", j === i);
      });
      counter.textContent = `Step ${i + 1} of ${frames.length}`;
      previous.disabled = i === 0;
      next.disabled = i === frames.length - 1;
      // A button disabled under the keyboard focus would drop it.
      if (document.activeElement?.disabled) {
        (previous.disabled ? next : previous).focus();
      }
    };
    previous.addEventListener("click", () => show(current - 1));
    next.addEventListener("click", () => show(current + 1));
    show(0);
  }
});
