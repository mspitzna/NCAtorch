// painting.js
export function setupPaintBrush(ctx, brushSize = 1, color = [0, 0, 0]) {
    let drawing = false;
  
    function draw(e) {
      const rect = canvas.getBoundingClientRect();
      const x = e.clientX - rect.left;
      const y = e.clientY - rect.top;
      ctx.fillStyle = `rgb(${color[0]}, ${color[1]}, ${color[2]})`;
      ctx.beginPath();
      ctx.arc(x, y, 1, 0, 2 * Math.PI);
      ctx.fill();
      sendPaintAction(x, y, brushSize, color);
    }
  
    canvas.addEventListener("mousedown", (e) => {
      drawing = true;
      draw(e);
    });
  
    canvas.addEventListener("mouseup", () => (drawing = false));
    canvas.addEventListener("mouseleave", () => (drawing = false));
    canvas.addEventListener("mousemove", throttle((e) => {
      if (drawing) draw(e);
    }, 50));
  }
  
  export function setupEraser(ctx, brushSize = 10) {
    let erasing = false;
  
    function erase(e) {
      const rect = canvas.getBoundingClientRect();
      const x = e.clientX - rect.left;
      const y = e.clientY - rect.top;
  
      ctx.globalCompositeOperation = 'destination-out'; // "erase" by painting transparency
      ctx.beginPath();
      ctx.arc(x, y, 1, 0, 2 * Math.PI);
      ctx.fill();
      ctx.globalCompositeOperation = 'source-over'; // Reset back to default drawing mode
      sendEraseAction(x, y, brushSize);
    }
  
    canvas.addEventListener("mousedown", (e) => {
      erasing = true;
      erase(e);
    });
  
    canvas.addEventListener("mouseup", () => (erasing = false));
    canvas.addEventListener("mouseleave", () => (erasing = false));
    canvas.addEventListener("mousemove", throttle((e) => {
      if (erasing) erase(e);
    }, 50));
  }
  
  function sendPaintAction(x, y, brushSize, color) {
    fetch("/paint", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ x, y, brush_size: brushSize, color }),
    })
    .catch((error) => console.error("Error sending paint action:", error));
  }
  
  function sendEraseAction(x, y, brushSize) {
    fetch("/erase", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ x, y, brush_size: brushSize }),
    })
    .catch((error) => console.error("Error sending erase action:", error));
  }
  
  function throttle(func, limit) {
    let lastFunc;
    let lastRan;
    return function () {
      const context = this;
      const args = arguments;
      if (!lastRan) {
        func.apply(context, args);
        lastRan = Date.now();
      } else {
        clearTimeout(lastFunc);
        lastFunc = setTimeout(function () {
          if (Date.now() - lastRan >= limit) {
            func.apply(context, args);
            lastRan = Date.now();
          }
        }, limit - (Date.now() - lastRan));
      }
    };
  }
  
export function setupSeedDot(ctx) {
  function placeSeedDot(e) {
    const rect = canvas.getBoundingClientRect();
    const x = Math.floor(e.clientX - rect.left);
    const y = Math.floor(e.clientY - rect.top);
    sendSeedDot(x, y);
  }

  canvas.addEventListener("dblclick", placeSeedDot);
}

function sendSeedDot(x, y) {
  fetch("/set_seed_dot", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ x, y }),
  })
  .catch((error) => console.error("Error sending seed dot:", error));
}
