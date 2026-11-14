import puppeteer from 'puppeteer';
import {spawn} from 'node:child_process';
import fs from 'node:fs';
const out='/downloads/evater-reviewed-prep/evidence';fs.mkdirSync(out,{recursive:true});
const server=spawn(process.execPath,['node_modules/vite/bin/vite.js','--host','127.0.0.1','--port','4181'],{stdio:['ignore','pipe','pipe']});
const prod=spawn(process.execPath,['node_modules/vite/bin/vite.js','preview','--host','127.0.0.1','--port','4182'],{stdio:['ignore','pipe','pipe']});
const browser=await puppeteer.launch({executablePath:'/opt/google/chrome/chrome',headless:true,args:['--no-sandbox']});const checks=[];
async function wait(url){for(let i=0;i<100;i++){try{const r=await fetch(url);if(r.ok)return;}catch{}await new Promise(r=>setTimeout(r,150));}throw Error('server unavailable');}
try{
 await wait('http://127.0.0.1:4181');await wait('http://127.0.0.1:4182');
 for(const width of [320,390,1280]){
  const page=await browser.newPage();await page.setViewport({width,height:900});await page.setRequestInterception(true);page.on('request',r=>{if(r.url().startsWith('http')&&!r.url().startsWith('http://127.0.0.1:'))r.abort();else r.continue();});let errors=[];page.on('pageerror',e=>errors.push(e.message));
  for(const slug of ['lenses','power-play','invisible-living-world']){
   await page.goto(`http://127.0.0.1:4181/reviewed-content/${slug}`,{waitUntil:'domcontentloaded'});await page.waitForSelector('article h1');
   const result=await page.evaluate(()=>({overflow:document.documentElement.scrollWidth>innerWidth,heading:document.querySelector('article h1').textContent,noindex:document.querySelector('meta[name=robots]')?.content,pending:document.body.innerText.includes('Owner publication decision pending.'),testLinks:[...document.querySelectorAll('article a')].filter(a=>a.pathname==='/practice').length,paragraphs:document.querySelectorAll('article p').length}));
   if(result.overflow||!result.noindex?.includes('noindex')||!result.pending||result.testLinks||result.paragraphs<8)throw Error(JSON.stringify(result));checks.push({width,slug,...result});
   await page.screenshot({path:`${out}/${width}-${slug}.png`,fullPage:true});
   await page.click('a[href="/reviewed-content/power-play"]');await page.waitForFunction(()=>location.pathname.endsWith('/power-play'));await page.waitForSelector('article h1');
   if(!await page.$eval('article h1',x=>x.textContent.includes('Power Play')))throw Error('navigation');
  }
  await page.goto('http://127.0.0.1:4181/reviewed-content/not-a-guide',{waitUntil:'domcontentloaded'});await page.waitForFunction(()=>document.body.innerText.includes('Preview not found'));if(!await page.evaluate(()=>document.body.innerText.includes('Preview not found')))throw Error('missing fallback');
  await page.goto('http://127.0.0.1:4182/reviewed-content/lenses',{waitUntil:'domcontentloaded'});await page.waitForFunction(()=>document.body.innerText.includes('Page not found'));if(await page.evaluate(()=>document.body.innerText.includes('predict the image before you look')||document.body.innerText.includes('Review preview - not published')))throw Error('production leak');checks.push({width,productionRouteExcluded:true,errors});if(errors.length)throw Error(errors.join('\n'));await page.screenshot({path:`${out}/${width}-production-excluded.png`,fullPage:true});await page.close();
 }
 fs.writeFileSync(`${out}/browser-checks.json`,JSON.stringify({passed:true,checks},null,2));console.log('PASS9guideviews3widths;navigation/unknownfallback/productionexclusion;0pageerrors');
}finally{await browser.close();server.kill();prod.kill();}
