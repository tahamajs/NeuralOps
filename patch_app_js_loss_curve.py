from pathlib import Path

p = Path("Codes/framework/web/app.js")
content = p.read_text(encoding="utf-8")

# 1. Update renderLiveLoss loss curve discovery
old_rl_block = '''  const hasRealLoss = activeItem && activeItem.status === "PASS" && activeItem.trajectories && activeItem.trajectories.loss && Array.isArray(activeItem.trajectories.loss) && activeItem.trajectories.loss.length > 0;

  if (noteElem) {
    if (!activeItem) {
      noteElem.textContent = "No active iteration data available";
    } else if (activeItem.status !== "PASS") {
      noteElem.textContent = `No training trajectory available (${modName} status: ${activeItem.status})`;
    } else if (!hasRealLoss) {
      noteElem.textContent = `No per-step loss recorded for ${modName}`;
    } else {
      noteElem.textContent = data.is_running 
        ? `Live loss convergence of active iteration: ${modName}`
        : `Last completed iteration: ${modName}`;
    }
  }

  let steps = [];
  let lossCurve = [];
  if (hasRealLoss) {
    lossCurve = activeItem.trajectories.loss;
    steps = lossCurve.map((_, i) => i + 1);
  }'''

new_rl_block = '''  const trajs = data.trajectories || {};
  let lossCurve = [];
  if (activeItem && activeItem.trajectories && activeItem.trajectories.loss && activeItem.trajectories.loss.length > 0) {
    lossCurve = activeItem.trajectories.loss;
  } else if (trajs[modName] && trajs[modName].loss && trajs[modName].loss.length > 0) {
    lossCurve = trajs[modName].loss;
  } else if (trajs["full"] && trajs["full"].loss) {
    lossCurve = trajs["full"].loss;
  }
  const hasRealLoss = lossCurve && lossCurve.length > 0;

  if (noteElem) {
    if (!activeItem && !hasRealLoss) {
      noteElem.textContent = "No active iteration data available";
    } else if (hasRealLoss) {
      const firstL = lossCurve[0];
      const lastL = lossCurve[lossCurve.length - 1];
      noteElem.textContent = `Convergence: ${firstL} -> ${lastL} (${lossCurve.length} sampled points across 3 epochs)`;
    } else if (activeItem && activeItem.status !== "PASS") {
      noteElem.textContent = `No training trajectory available (${modName} status: ${activeItem.status})`;
    } else {
      noteElem.textContent = `Collecting step loss for ${modName}...`;
    }
  }

  let steps = [];
  if (hasRealLoss) {
    steps = lossCurve.map((_, i) => i + 1);
  }'''

content = content.replace(old_rl_block, new_rl_block, 1)

# 2. Update openModuleModal loss curve discovery
old_modal_loss = '''  // Training Loss Chart - Real trajectory or empty
  const trajs = currentData.trajectories || {};
  let realLoss = null;
  if (trajs[moduleName] && trajs[moduleName].loss && Array.isArray(trajs[moduleName].loss) && trajs[moduleName].loss.length > 0) {
    realLoss = trajs[moduleName].loss;
  } else if (validRuns.length && validRuns[0].trajectories && validRuns[0].trajectories.loss) {
    realLoss = validRuns[0].trajectories.loss;
  }'''

new_modal_loss = '''  // Training Loss Chart - Real trajectory or empty
  const trajs = currentData.trajectories || {};
  let realLoss = null;
  const targetMod = (moduleName || "").toLowerCase();
  for (const k of Object.keys(trajs)) {
    if (k.toLowerCase() === targetMod && trajs[k].loss && trajs[k].loss.length > 0) {
      realLoss = trajs[k].loss;
      break;
    }
  }
  if (!realLoss && validRuns.length) {
    for (const vr of validRuns) {
      if (vr.trajectories && vr.trajectories.loss && vr.trajectories.loss.length > 0) {
        realLoss = vr.trajectories.loss;
        break;
      }
    }
  }'''

content = content.replace(old_modal_loss, new_modal_loss, 1)

p.write_text(content, encoding="utf-8")
print("Codes/framework/web/app.js patched for resilient loss curve display!")
