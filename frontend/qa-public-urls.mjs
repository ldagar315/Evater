import puppeteer from 'puppeteer'
import { spawn } from 'node:child_process'
import { readFile } from 'node:fs/promises'
import assert from 'node:assert/strict'
const server=spawn('python3',['-m','http.server','4181','--bind','127.0.0.1','--directory','dist'],{stdio:'ignore'})
const wait=ms=>new Promise(r=>setTimeout(r,ms));let browser
try {
 for(let i=0;i<80;i++){try{await fetch('http://127.0.0.1:4181');break}catch{await wait(50)}}
 const urls=[...(await readFile('dist/sitemap.xml','utf8')).matchAll(/<loc>(.*?)<\/loc>/g)].map(m=>m[1])
 assert.equal(urls.length,13);assert.equal(new Set(urls).size,13)
 for(const url of urls){
  const path=new URL(url).pathname;const r=await fetch('http://127.0.0.1:4181'+path,{redirect:'manual'});assert.equal(r.status,200,url)
  const html=await r.text();assert(html.includes(`rel="canonical" href="${url}"`),`canonical ${url}`);assert(html.includes(`property="og:url" content="${url}"`),`og:url ${url}`)
  for (const image of html.matchAll(/src="(\/blog\/diagrams\/[^"]+)"/g)) { const r=await fetch('http://127.0.0.1:4181'+image[1],{redirect:'manual'}); assert.equal(r.status,200, image[1]); assert(r.headers.get('content-type')?.includes('svg'),image[1]) }
  assert(!/href="\/blog(?:\?|"|\/[^"?#]+[^/]")/.test(html),'no redirecting public links in '+url)
 }
 browser=await puppeteer.launch({executablePath:'/usr/bin/google-chrome',args:['--no-sandbox','--disable-dev-shm-usage']})
 const page=await browser.newPage();await page.setViewport({width:1440,height:1000})
 // Block all external APIs: no account or auth state involved.
 await page.setRequestInterception(true);page.on('request',r=>r.url().startsWith('http://127.0.0.1:4181')||r.url().startsWith('data:')||r.resourceType()==='image'?r.continue():r.abort())
 await page.goto('http://127.0.0.1:4181/about/#privacy');await page.waitForFunction(()=>document.querySelector('link[rel=canonical]')?.href==='https://evater.xyz/about/');assert(await page.$('a[aria-current=page][href="/about/"]'));await page.screenshot({path:'/downloads/evater-public-about-slash.png'})
 await page.goto('http://127.0.0.1:4181/blog/');await page.waitForSelector('a[href^="/blog/"][href$="/"]');await page.screenshot({path:'/downloads/evater-public-blog-slash.png'})
 await page.setViewport({width:390,height:844,isMobile:true});await page.goto('http://127.0.0.1:4181/blog/active-recall-class-6-10-exams/');await page.waitForFunction(()=>document.querySelector('link[rel=canonical]')?.href.endsWith('/active-recall-class-6-10-exams/'));assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth),false);await page.screenshot({path:'/downloads/evater-public-article-slash-mobile.png'})
 console.log('PASS13 sitemap targets served directly200 by directory-index server, matching canonical+og:url, no duplicate URLs, public links slash-aligned; About hash/current nav and desktop/mobile rendered checks. Not production deployment/indexing proof.')
} finally {if(browser)await browser.close();server.kill()}
