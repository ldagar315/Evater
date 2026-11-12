import puppeteer from 'puppeteer';
import fs from 'node:fs';
import path from 'node:path';
import {spawn} from 'node:child_process';
import assert from 'node:assert/strict';
const root=path.resolve(import.meta.dirname,'../..');
const evidence=process.env.REVIEW_EVIDENCE;
assert(evidence && process.env.REVIEW_INPUTS && process.env.REVIEW_PYTHON,'Set REVIEW_EVIDENCE, REVIEW_INPUTS, REVIEW_PYTHON');
fs.mkdirSync(evidence,{recursive:true});
const log=fs.openSync(path.join(evidence,'server.log'),'w');
const server=spawn(process.env.REVIEW_PYTHON,['-m','review_tools.serve_mixed_review','--inputs',process.env.REVIEW_INPUTS,'--port','8766'],{cwd:root,env:{...process.env,PYTHONPATH:'backend'},stdio:['ignore',log,log]});
let browser;
const traces=[];const results=[];const failures=[];
try {
  let ready=false;for(let i=0;i<100;i++){try{const r=await fetch('http://127.0.0.1:8766/');if(r.ok){ready=true;break}}catch{} await new Promise(r=>setTimeout(r,100));}
  assert(ready,'local server did not become ready');
  browser=await puppeteer.launch({executablePath:'/usr/bin/google-chrome',headless:true,args:['--no-sandbox','--disable-dev-shm-usage']});
  for(const width of [320,390,1280]){
    const page=await browser.newPage();await page.setViewport({width,height:900});
    page.on('pageerror',e=>failures.push(String(e)));
    page.on('response',async response=>{if(response.url().includes('/review/mixed-sample-papers/')){traces.push({width,url:response.url(),status:response.status(),body:await response.json()});}});
    await page.goto('http://127.0.0.1:8766/',{waitUntil:'networkidle0'});
    await page.waitForSelector('.mixed-review');
    assert.equal(await page.$$eval('main>ol',els=>els.map(e=>e.children.length).join(',')),'10,5');
    assert.equal(await page.$eval('body',e=>e.scrollWidth<=window.innerWidth),true);
    await page.screenshot({path:path.join(evidence,`synthetic-${width}-top.png`)});
    await page.$eval('main>ol:nth-of-type(2)',e=>e.scrollIntoView({block:'center'}));
    await page.screenshot({path:path.join(evidence,`synthetic-${width}-written.png`)});
    await page.$$eval('details',els=>els.forEach(e=>e.open=true));
    await page.$eval('details:last-child',e=>e.scrollIntoView({block:'end'}));
    assert(await page.$eval('details:last-child',e=>e.open && e.innerText.includes('Synthetic answer for wiring only.')));
    await page.screenshot({path:path.join(evidence,`synthetic-${width}-answers.png`)});
    results.push({width,case:'synthetic HTTP/component/disclosures/overflow',pass:true});
    for(const id of ['light','electricity','maths','unknown','broken']){
      await page.select('select',id);
      await page.waitForSelector('[role=alert]');
      assert.equal(await page.$('main'),null,'no stale synthetic paper after failure');
      const text=await page.$eval('[role=alert]',e=>e.innerText);
      if(id==='light')assert(text.includes('Q3: bank answer conflicts with reviewed answer'));
      if(['light','electricity','maths'].includes(id)){
        if(process.env.REVIEW_FRESH_BANK==='1'){assert(!text.includes('current bank state is not verified live'));assert(!text.includes('bank check is stale'));assert(!text.includes('bank check time is missing or invalid'));}
        else assert(text.includes('current bank state is not verified live'));
      }
      if(id==='unknown')assert(text.includes('404'));
      if(id==='broken')assert(text.includes('review loader failed'));
      assert.equal(await page.$eval('body',e=>e.scrollWidth<=window.innerWidth),true);
      if(id==='light'){await page.evaluate(()=>window.scrollTo(0,0));await page.screenshot({path:path.join(evidence,`held-light-${width}.png`),fullPage:true});}
      results.push({width,case:id+' failure clears old paper',pass:true});
    }
    await page.reload({waitUntil:'networkidle0'});await page.waitForSelector('main');
    results.push({width,case:'direct reload synthetic HTTP',pass:true});
    // Transport failure after a successful paper must not leave that paper visible.
    await page.setRequestInterception(true);
    page.on('request',request=>request.url().includes('/review/mixed-sample-papers/')?request.abort('failed'):request.continue());
    await page.select('select','light');await page.waitForSelector('[role=alert]');
    assert.equal(await page.$('main'),null);
    assert((await page.$eval('[role=alert]',e=>e.innerText)).includes('Review request failed'));
    results.push({width,case:'network failure clears old paper',pass:true});
    await page.close();
  }
  assert.equal(failures.length,0,'browser runtime errors');
} finally {
  if(browser)await browser.close();server.kill('SIGTERM');
  await new Promise(resolve=>server.once('exit',resolve));fs.closeSync(log);
  fs.writeFileSync(path.join(evidence,'http-traces.json'),JSON.stringify(traces,null,2));
  fs.writeFileSync(path.join(evidence,'browser-results.json'),JSON.stringify({results,failures},null,2));
}
console.log(`${results.length} browser cases PASS; ${traces.length} actual HTTP response traces`);
