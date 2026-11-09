import puppeteer from 'puppeteer'
import { spawn } from 'node:child_process'
import assert from 'node:assert/strict'
const server=spawn('./node_modules/.bin/vite',['preview','--host','127.0.0.1','--port','4176'],{stdio:'ignore'})
const wait=ms=>new Promise(r=>setTimeout(r,ms))
for(let i=0;i<60;i++){try{await fetch('http://127.0.0.1:4176');break}catch{await wait(100)}}
const browser=await puppeteer.launch({executablePath:process.env.CHROME_PATH||'/usr/bin/google-chrome',args:['--no-sandbox','--disable-dev-shm-usage']})
const user={id:'00000000-0000-4000-8000-000000000099',aud:'authenticated',role:'authenticated',email:'qa@example.invalid',app_metadata:{provider:'email'},user_metadata:{},created_at:'2026-01-01T00:00:00Z'}
const profile={id:'mock-profile',created_by:user.id,user_name:'QA Learner',name:'QA Learner',email:user.email,grade:8,school:'Existing custom campus',credits:100}
const token=`${Buffer.from(JSON.stringify({alg:'HS256',typ:'JWT'})).toString('base64url')}.${Buffer.from(JSON.stringify({sub:user.id,exp:Math.floor(Date.now()/1000)+3600})).toString('base64url')}.mock`
async function setup({mobile=false,empty=false,fail=false}={}) {
 const p=await browser.newPage();await p.setViewport(mobile?{width:390,height:844,isMobile:true,hasTouch:true,deviceScaleFactor:1}:{width:1440,height:1000});
 const reads=[];let failures=fail?1:0
 await p.evaluateOnNewDocument((session)=>localStorage.setItem('sb-qfxrfwgipsygoozlnkwb-auth-token',JSON.stringify(session)),{access_token:token,refresh_token:'mock-refresh',expires_in:3600,expires_at:Math.floor(Date.now()/1000)+3600,token_type:'bearer',user})
 await p.setRequestInterception(true);p.on('request',async req=>{
  const u=new URL(req.url());if(u.hostname==='127.0.0.1'||u.protocol==='data:')return req.continue()
  const send=(status,data)=>req.respond({status,contentType:'application/json',headers:{'access-control-allow-origin':'*','access-control-allow-headers':'*'},body:JSON.stringify(data)})
  if(req.method()==='OPTIONS')return send(200,{})
  if(u.pathname==='/auth/v1/user')return send(200,user)
  if(u.pathname==='/rest/v1/Users')return send(200,profile)
  if(u.pathname==='/rest/v1/test_attempts'){
   reads.push({method:req.method(),url:u.href});await wait(250)
   if(failures-->0)return send(503,{message:'mock unavailable'})
   if(empty)return send(200,[])
   const offset=Number(u.searchParams.get('offset')||0)
   return send(200,offset?[{id:'older',status:'completed',mode:'review',score:10,maximum_marks:10,started_at:'2026-11-01T10:00:00Z',completed_at:'2026-11-01T10:10:00Z',chapters:null}]:Array.from({length:21},(_,i)=>({id:String(i),status:i===1?'in_progress':'completed',mode:i===2?'review':'practice',score:i===1?null:6,maximum_marks:i===1?null:10,started_at:'2026-11-08T10:00:00Z',completed_at:i===1?null:'2026-11-08T10:10:00Z',chapters:i===2?null:{title:'Fractions in Disguise',curriculum_versions:{subject:'Mathematics'}}})))
  }
  if(u.pathname==='/rest/v1/Questions_Created')return send(200,[{id:123,created_at:'2026-10-01T10:00:00Z',subject:'Science',grade:'8',chapter:'Light',difficulty_level:'easy',test:{questions:[]}}])
  return send(200,[])
 });return {p,reads}
}
const base='http://127.0.0.1:4176'
try {
 const desktop=await setup();await desktop.p.goto(base+'/home');await desktop.p.waitForFunction(()=>Array.from(document.querySelectorAll('button')).some(b=>b.textContent.trim()==='My tests'));await desktop.p.evaluate(()=>Array.from(document.querySelectorAll('button')).find(b=>b.textContent.trim()==='My tests').click());await desktop.p.waitForSelector('#practice-history-title');await desktop.p.waitForFunction(()=>document.body.innerText.includes('Fractions in Disguise'));assert.match(await desktop.p.$eval('main',e=>e.innerText),/6 \/ 10/);assert.match(await desktop.p.$eval('main',e=>e.innerText),/Not completed/);assert.match(await desktop.p.$eval('main',e=>e.innerText),/Weak-spot review/);assert.equal(await desktop.p.$$eval('section li',e=>e.length),20);await desktop.p.screenshot({path:'/downloads/evater-my-tests-desktop.png',fullPage:false});await desktop.p.evaluate(()=>Array.from(document.querySelectorAll('button')).find(b=>b.textContent.includes('Show older')).click());await desktop.p.waitForFunction(()=>document.querySelectorAll('section li').length===21);assert.equal(desktop.reads.length,2);assert(new URL(desktop.reads[1].url).searchParams.get('offset')==='20');assert(desktop.reads.every(r=>r.method==='GET'&&new URL(r.url).searchParams.get('user_id')==='eq.'+user.id));await desktop.p.click('a[href="/previous-tests"]');await desktop.p.waitForFunction(()=>document.body.innerText.includes('Light'));assert(desktop.p.url().endsWith('/previous-tests'));await desktop.p.close()
 const mobile=await setup({mobile:true});await mobile.p.goto(base+'/home');await mobile.p.waitForSelector('button[aria-label="Open navigation menu"]');await mobile.p.click('button[aria-label="Open navigation menu"]');const nav=await mobile.p.$('header .absolute nav[aria-label="Learning navigation"] a[href="/my-tests"]');await nav.tap();await mobile.p.waitForFunction(()=>document.body.innerText.includes('Fractions in Disguise'));assert.equal(await mobile.p.$$('button[aria-label="Close navigation menu"]').then(x=>x.length),0);assert.equal(await mobile.p.evaluate(()=>document.documentElement.scrollWidth>innerWidth),false);await mobile.p.screenshot({path:'/downloads/evater-my-tests-mobile.png'});await mobile.p.close()
 const error=await setup({fail:true});await error.p.goto(base+'/my-tests');await error.p.waitForSelector('[role=alert]');assert(!await error.p.$eval('main',e=>e.innerText.includes('No practice tests yet')));await error.p.evaluate(()=>Array.from(document.querySelectorAll('button')).find(b=>b.textContent.includes('Retry history')).click());await error.p.waitForFunction(()=>document.body.innerText.includes('Fractions in Disguise'));await error.p.close()
 const empty=await setup({empty:true,mobile:true});await empty.p.goto(base+'/my-tests');await empty.p.waitForFunction(()=>document.body.innerText.includes('No practice tests yet'));assert(await empty.p.$('a[href="/previous-tests"]'));assert(await empty.p.$('a[href="/previous-feedbacks"]'));await empty.p.close()
 console.log('PASS: home CTA, desktop/mobile menu discovery, completed score, unfinished label, cross-chapter review,20-row pagination, owned GET-only query, paper-history link, error retry and empty state. All remote requests mocked. No live writes.')
} finally {await browser.close();server.kill()}
