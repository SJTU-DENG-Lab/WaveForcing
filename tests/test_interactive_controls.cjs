// Real browser test; run `python tests/interactive_synthetic_server.py` (manual start) or set WAVERT_TEST_URL.
const assert=require('node:assert/strict'),fs=require('node:fs'),path=require('node:path');
const {chromium}=require(process.env.PLAYWRIGHT_MODULE||'playwright');
(async()=>{
 const browser=await chromium.launch({headless:true,args:['--no-sandbox']});const page=await browser.newPage();const errors=[];const out=process.env.WAVERT_TEST_OUTPUT;
 page.on('pageerror',e=>errors.push(e.message));
 try{
  await page.goto(process.env.WAVERT_TEST_URL||'http://127.0.0.1:8892/');
  await page.waitForFunction(()=>!document.querySelector('#start').disabled);
  assert.equal(await page.locator('#start').textContent(),'开始生成');assert.equal(await page.locator('#initial-prompt').inputValue(),'');assert(await page.locator('#change').isDisabled());
  await page.waitForFunction(()=>document.querySelectorAll('.gpu').length===8);assert((await page.locator('#memory-summary').textContent()).includes('GiB'));
  const readState=()=>page.evaluate(()=>api('/state'));
  const idle=await readState();assert.equal(idle.has_session,false);assert.equal(idle.end,0);
  const diag=()=>page.evaluate(()=>window.wavertDiagnostics());
  const color=()=>page.evaluate(()=>{const v=document.querySelector('video'),c=document.createElement('canvas');c.width=1;c.height=1;const ctx=c.getContext('2d');ctx.drawImage(v,700,400,1,1,0,0,1,1);const rgb=Array.from(ctx.getImageData(0,0,1,1).data).slice(0,3);return rgb.indexOf(Math.max(...rgb));});
  await page.fill('#initial-prompt','A red beginning');await page.click('#start');await page.waitForFunction(()=>window.wavertDiagnostics().started);await page.waitForTimeout(1200);
  const first=await readState(),d1=await diag();assert.equal(first.initial_prompt,'A red beginning');assert.equal(await color(),0);assert.equal(await page.locator('#start').textContent(),'重新生成');
  // Regenerate while a requested prompt change is still on its way to the screen.
  await page.fill('#prompt','Later green');await page.click('#change');await page.waitForFunction(()=>window.wavertTrace.recent().some(r=>r.event==='api_end'&&r.data.path==='/prompt'&&r.data.status===200),{},{timeout:15000});
  await page.fill('#initial-prompt','A new beginning');await page.click('#start');
  await page.waitForFunction(id=>window.wavertDiagnostics().session_id!==id&&window.wavertDiagnostics().started,first.session_id);
  // The synthetic source advances one colour per request: start, prompt change, restart.
  await page.waitForTimeout(2400);const second=await readState(),d2=await diag();assert.equal(second.initial_prompt,'A new beginning');assert.equal(d2.branch,0);assert.equal(d2.dead,false);assert.equal(d2.underruns,0);assert.equal(await color(),2);
  const stale=await page.evaluate(async id=>{const headers={'Content-Type':'application/json','X-Client-Id':token,'X-Session-Id':id};return Promise.all([fetch('played',{method:'POST',headers,body:JSON.stringify({branch:0,chunk:999})}),fetch('chunk/0/0',{headers})].map(async r=>(await r).status));},first.session_id);assert.deepEqual(stale,[409,409]);
  // Same prompt, a fresh seed, and a new timeline for another regeneration.
  await page.click('#start');await page.waitForFunction(id=>window.wavertDiagnostics().session_id!==id&&window.wavertDiagnostics().started,second.session_id);await page.waitForTimeout(1300);
  const third=await readState(),d3=await diag();assert.notEqual(third.seed,second.seed);assert.equal(third.initial_prompt,second.initial_prompt);assert.equal(await color(),0);assert(d3.currentTime<3);assert.equal(d3.underruns,0);
  await page.click('#pause');const paused=await diag(),metricsBefore=await page.locator('#memory-summary').textContent();await page.waitForTimeout(2100);const after=await diag();assert(Math.abs(after.currentTime-paused.currentTime)<.1);assert(after.acked-paused.acked<=1);assert.notEqual(await page.locator('#memory-summary').textContent(),metricsBefore);
  await page.click('#pause');await page.waitForTimeout(1000);const beforeReload=await diag();await page.reload();await page.waitForFunction(()=>window.wavertDiagnostics().started);await page.waitForTimeout(1300);const resumed=await diag();assert.equal(resumed.session_id,third.session_id);assert(resumed.acked>=beforeReload.acked);assert.equal(resumed.dead,false);assert.equal(await page.locator('#initial-prompt').inputValue(),third.initial_prompt);assert.equal(await color(),0);
  assert.deepEqual(errors,[]);
  const report={idle,first,second,third,diagnostics:[d1,d2,d3,resumed],stale,errors,gpu_cards:await page.locator('.gpu').count()};
  if(out){await page.screenshot({path:path.join(out,'controls_synthetic.png'),fullPage:true});fs.writeFileSync(path.join(out,'browser_controls.json'),JSON.stringify(report,null,2)+'\n');}
  console.log(JSON.stringify(report,null,2));
 }finally{await browser.close();}
})().catch(e=>{console.error(e);process.exit(1);});
