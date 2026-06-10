// painting.js
export function setupPaintBrush(ctx, brushSize = 1, color = [0, 0, 0]) {
    let drawing = false;
    const getSize = typeof brushSize === 'function' ? brushSize : () => brushSize;

    function draw(e) {
      const rect = canvas.getBoundingClientRect();
      const x = e.clientX - rect.left;
      const y = e.clientY - rect.top;
      ctx.fillStyle = `rgb(${color[0]}, ${color[1]}, ${color[2]})`;
      ctx.fillRect(x - 1, y - 1, 2, 2);
      sendPaintAction(x, y, getSize(), color);
    }

    canvas.addEventListener("mousedown", (e) => {
      drawing = true;
      draw(e);
    });

    window.addEventListener("mouseup", () => (drawing = false));
    window.addEventListener("mousemove", throttle((e) => {
      if (drawing) draw(e);
    }, 16));
  }
  
export function setupEraser(ctx, brushSize = 10) {
    let erasing = false;

    function erase(e) {
      const rect = canvas.getBoundingClientRect();
      const x = e.clientX - rect.left;
      const y = e.clientY - rect.top;

      ctx.globalCompositeOperation = 'destination-out';
      ctx.beginPath();
      ctx.arc(x, y, 1, 0, 2 * Math.PI);
      ctx.fill();
      ctx.globalCompositeOperation = 'source-over';
      sendEraseAction(x, y, brushSize);
    }

    canvas.addEventListener("mousedown", (e) => {
      erasing = true;
      erase(e);
    });

    window.addEventListener("mouseup", () => (erasing = false));
    window.addEventListener("mousemove", throttle((e) => {
      if (erasing) erase(e);
    }, 16));
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
    let lastRan = 0;
    return function () {
      const now = Date.now();
      if (now - lastRan >= limit) {
        lastRan = now;
        func.apply(this, arguments);
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
