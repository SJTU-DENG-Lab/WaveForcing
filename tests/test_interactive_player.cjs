// Real browser test; run `python tests/interactive_synthetic_server.py --autostart` or set WAVERT_TEST_URL.
const assert=require('node:assert/strict');
const {chromium}=require(process.env.PLAYWRIGHT_MODULE||'playwright');
(async()=>{
 const browser=await chromium.launch({headless:true,args:['--no-sandbox']});
 const page=await browser.newPage();const errors=[];
 page.on('pageerror',e=>errors.push(e.message));
 await page.goto(process.env.WAVERT_TEST_URL||'http://127.0.0.1:8892/');
 await page.waitForFunction(()=>window.wavertDiagnostics?.().started,{},{timeout:30000});
 await page.evaluate(()=>{
  window.decodedSamples=[];const v=document.querySelector('video'),c=document.createElement('canvas');c.width=1;c.height=1;const ctx=c.getContext('2d');
  const sample=(now,m)=>{ctx.drawImage(v,700,400,1,1,0,0,1,1);const rgb=Array.from(ctx.getImageData(0,0,1,1).data.slice(0,3));window.decodedSamples.push({t:m.mediaTime,wall:performance.now(),colour:rgb.indexOf(Math.max(...rgb))});v.requestVideoFrameCallback(sample);};v.requestVideoFrameCallback(sample);
 });
 const snapshot=()=>page.evaluate(()=>window.wavertDiagnostics());
 // A prompt change plays right behind the chunks committed before it, on the same timeline.
 const change=async(text,colour)=>{
  await page.fill('#prompt',text);const clicked=await page.evaluate(()=>performance.now());await page.click('#change');
  await page.waitForFunction(c=>window.decodedSamples.at(-1)?.colour===c,colour,{timeout:15000});
  return await page.evaluate(c=>window.decodedSamples.find(s=>s.colour===c).wall,colour)-clicked;
 };
 await page.waitForTimeout(4000);
 const delays=[await change('green',1)];
 await page.waitForTimeout(6500);
 delays.push(await change('blue',2));
 await page.waitForTimeout(7500);
 const beforePause=await snapshot();await page.click('#pause');await page.waitForTimeout(1500);
 const stopped=await snapshot();assert(stopped.currentTime-beforePause.currentTime<.3);assert(stopped.acked-beforePause.acked<=1);
 await page.click('#pause');await page.waitForTimeout(10000);
 const result=await snapshot();const samples=await page.evaluate(()=>window.decodedSamples);
 assert.equal(result.dead,false,JSON.stringify(result));assert.equal(result.underruns,0,JSON.stringify(result));assert.equal(result.switches,0);
 assert(result.acked>30);assert(result.cacheEntries<40);assert.equal(errors.length,0,errors.join('\n'));
 // The buffer is three chunks: a change must show within roughly that much playback.
 for(const delay of delays)assert(delay>0&&delay<4000,`Prompt change took ${delay} ms`);
 const colours=samples.map(s=>s.colour);
 console.log(JSON.stringify({result,samples:samples.length,colours:[...new Set(colours)],prompt_change_ms:delays,max_frame_gap:Math.max(...samples.slice(1).map((s,i)=>s.t-samples[i].t)),errors},null,2));
 console.log('frame gaps',JSON.stringify(samples.slice(1).map((s,i)=>({at:s.t,gap:s.t-samples[i].t})).filter(x=>x.gap>.2)));
 for(const c of [0,1,2])assert(colours.includes(c),`Scene ${c} was not decoded`);
 for(let i=1;i<samples.length;i++){assert(samples[i].t>samples[i-1].t);assert(samples[i].t-samples[i-1].t<.2);assert(colours[i]>=colours[i-1],'Old scene reappeared after the change');}
 // Reload resumes at the last played chunk using a fresh MSE time origin.
 await page.reload();await page.waitForFunction(()=>window.wavertDiagnostics?.().started,{},{timeout:30000});await page.waitForTimeout(2000);
 const resumed=await snapshot();assert(resumed.acked>=result.acked);assert.equal(resumed.dead,false);assert.equal(resumed.underruns,0);
 console.log('Reload passed',JSON.stringify(resumed));await browser.close();
})().catch(e=>{console.error(e);process.exit(1);});
