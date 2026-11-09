import puppeteer from 'puppeteer'
import { spawn } from 'node:child_process'
import assert from 'node:assert/strict'
const server=spawn('./node_modules/.bin/vite',['preview','--host','127.0.0.1','--port','4174'],{stdio:'ignore'})
const wait=ms=>new Promise(r=>setTimeout(r,ms))
for(let i=0;i<60;i++){try{await fetch('http://127.0.0.1:4174');break}catch{await wait(100)}}
const browser=await puppeteer.launch({executablePath:process.env.CHROME_PATH||'/usr/bin/google-chrome',args:['--no-sandbox','--disable-dev-shm-usage']})
const user={id:'00000000-0000-4000-8000-000000000099',aud:'authenticated',role:'authenticated',email:'qa@example.invalid',app_metadata:{provider:'email'},user_metadata:{},created_at:'2026-01-01T00:00:00Z'}
const profile={id:'mock-profile',created_by:user.id,user_name:'QA Learner',name:'QA Learner',email:user.email,grade:8,school:'Existing custom campus',credits:100}
const token=`${Buffer.from(JSON.stringify({alg:'HS256',typ:'JWT'})).toString('base64url')}.${Buffer.from(JSON.stringify({sub:user.id,exp:Math.floor(Date.now()/1000)+3600})).toString('base64url')}.mock`
async function setup({existing=true,save='ok',mobile=false,chapterFailure=false}={}){
 const p=await browser.newPage();await p.setViewport(mobile?{width:390,height:844,isMobile:true,hasTouch:true,deviceScaleFactor:1}:{width:1440,height:1000});
 let stored=existing?{...profile}:null;const writes=[];let chapterCalls=0
 await p.evaluateOnNewDocument((session)=>localStorage.setItem('sb-qfxrfwgipsygoozlnkwb-auth-token',JSON.stringify(session)),{access_token:token,refresh_token:'mock-refresh',expires_in:3600,expires_at:Math.floor(Date.now()/1000)+3600,token_type:'bearer',user})
 await p.setRequestInterception(true);p.on('request',async req=>{
  const u=new URL(req.url());if(u.hostname==='127.0.0.1'||u.protocol==='data:')return req.continue()
  const send=(status,data)=>req.respond({status,contentType:'application/json',headers:{'access-control-allow-origin':'*','access-control-allow-headers':'*','access-control-allow-methods':'*'},body:JSON.stringify(data)})
  if(req.method()==='OPTIONS')return send(200,{})
  if(u.pathname==='/auth/v1/user')return send(200,user)
  if(u.pathname==='/rest/v1/Users'){
   if(req.method()==='GET')return stored?send(200,stored):send(406,{code:'PGRST116',message:'No rows'})
   writes.push({method:req.method(),body:JSON.parse(req.postData())});await wait(350)
   if(save==='error')return send(400,{message:'Mock save failed'})
   if(save==='empty')return send(200,null)
   stored={...profile,...(Array.isArray(writes.at(-1).body)?writes.at(-1).body[0]:writes.at(-1).body)};return send(200,stored)
  }
  if(u.pathname==='/api/v1/chapters'){
   chapterCalls++;await wait(700)
   if(chapterFailure&&chapterCalls===1)return send(503,{detail:'Mock temporary chapter failure'})
   return send(200,{chapters:[{id:'22222222-2222-4222-8222-222222222222',sequence_number:1,title:`${u.searchParams.get('subject')} chapter one`,description:'Mock published chapter'}]})
  }
  if(u.hostname.endsWith('supabase.co'))return send(200,[])
  return send(200,{blocks:[],chapters:[],entries:[],data:[]})
 });return {p,writes,get chapterCalls(){return chapterCalls}}
}
const base='http://127.0.0.1:4174'
try{
 // Existing custom data survives initial load, canonical selection requires an actual option.
 const x=await setup();await x.p.goto(base+'/profile');await x.p.waitForSelector('input[autocomplete="organization"]');assert.equal(await x.p.$eval('input[autocomplete="organization"]',e=>e.value),profile.school)
 await x.p.evaluate(()=>Array.from(document.querySelectorAll('button')).find(b=>b.textContent.includes('Search the Gurugram')).click())
 await x.p.type('[role=combobox]','euro 45');await x.p.waitForSelector('[role=option]');assert.equal(await x.p.$eval('button[type=submit]',e=>e.disabled),true)
 await x.p.keyboard.press('ArrowDown');await x.p.keyboard.press('Enter');assert.match(await x.p.$eval('[role=combobox]',e=>e.value),/SECTOR-45/i)
 await x.p.screenshot({path:'/downloads/evater-profile-school-desktop.png',fullPage:true});await x.p.click('button[type=submit]');assert(x.p.url().endsWith('/profile'));await x.p.waitForFunction(()=>location.pathname==='/home');assert.equal(x.writes.length,1);assert.equal(x.writes[0].method,'PATCH');assert.match(x.writes[0].body.school,/SECTOR-45/i);await x.p.close()
 // Touch selection on a narrow viewport.
 const m=await setup({mobile:true});await m.p.goto(base+'/profile');await m.p.waitForSelector('input[autocomplete="organization"]');await m.p.evaluate(()=>Array.from(document.querySelectorAll('button')).find(b=>b.textContent.includes('Search the Gurugram')).click());await m.p.type('[role=combobox]','euro');await m.p.$eval('[role=combobox]',e=>e.scrollIntoView({block:'center'}));await m.p.screenshot({path:'/downloads/evater-school-mobile-search.png',fullPage:false});const first=await m.p.$('[role=option]');await first.tap();assert.equal(await m.p.$eval('[role=combobox]',e=>e.getAttribute('aria-expanded')),'false');assert.match(await m.p.$eval('[role=combobox]',e=>e.value),/EURO/);assert.equal(await m.p.evaluate(()=>document.documentElement.scrollWidth>innerWidth),false)
 // Reopen suggestions, physically swipe the inner list, choose Other, then reach Save after closing.
 await m.p.click('[role=combobox]',{clickCount:3});await m.p.keyboard.press('Backspace');await m.p.type('[role=combobox]','school');await m.p.$eval('[role=combobox]',e=>e.scrollIntoView({block:'center'}));
 const list=await m.p.$('[role=listbox]');const box=await list.boundingBox();const cdp=await m.p.createCDPSession();const sx=box.x+box.width/2, sy=box.y+box.height-30;
 await cdp.send('Input.dispatchTouchEvent',{type:'touchStart',touchPoints:[{x:sx,y:sy}]});for(let i=1;i<=8;i++){await cdp.send('Input.dispatchTouchEvent',{type:'touchMove',touchPoints:[{x:sx,y:sy-i*24}]});await wait(20)}await cdp.send('Input.dispatchTouchEvent',{type:'touchEnd',touchPoints:[]});
 assert((await m.p.$eval('[role=listbox]',e=>e.scrollTop))>0,'touch swipe must scroll dropdown')
 // Move to the final option using the same touch-scrolling surface; no outer-page overflow.
 for(let j=0;j<10;j++){await cdp.send('Input.dispatchTouchEvent',{type:'touchStart',touchPoints:[{x:sx,y:sy}]});for(let i=1;i<=8;i++)await cdp.send('Input.dispatchTouchEvent',{type:'touchMove',touchPoints:[{x:sx,y:sy-i*24}]});await cdp.send('Input.dispatchTouchEvent',{type:'touchEnd',touchPoints:[]})}
 await m.p.$eval('[role=combobox]',e=>e.scrollIntoView({block:'center'}));await m.p.waitForFunction(()=>{const el=document.querySelector('[role=listbox]');return el&&el.scrollTop+el.clientHeight>=el.scrollHeight-2});await m.p.screenshot({path:'/downloads/evater-school-mobile-scrolled.png'});const options=await m.p.$$('[role=option]');await options.at(-1).tap();await m.p.waitForSelector('input[autocomplete=organization]');await m.p.type('input[autocomplete=organization]',' - custom QA campus');await m.p.$eval('button[type=submit]',e=>e.scrollIntoView({block:'center'}));assert.equal(await m.p.$eval('button[type=submit]',e=>e.disabled),false);assert.equal(await m.p.$$eval('[role=listbox]',e=>e.length),0);await m.p.screenshot({path:'/downloads/evater-school-mobile-save.png'});await (await m.p.$('button[type=submit]')).tap();await m.p.waitForFunction(()=>location.pathname==='/home');assert.equal(m.writes.length,1);await m.p.close()
 // New profile uses Other, preserves custom text, and redirects only after a persisted result.
 for(const save of ['ok','error','empty']){
  const n=await setup({existing:false,save});await n.p.goto(base+'/profile');await n.p.waitForSelector('[role=combobox]');await n.p.type('input[placeholder="Enter your full name"]','New QA Learner');await n.p.select('select','8');await n.p.type('[role=combobox]','Unlisted QA Campus');await n.p.keyboard.press('ArrowDown');await n.p.keyboard.press('Enter');assert.equal(await n.p.$eval('input[autocomplete="organization"]',e=>e.value),'Unlisted QA Campus');await n.p.click('button[type=submit]');assert(n.p.url().endsWith('/profile'))
  if(save==='ok')await n.p.waitForFunction(()=>location.pathname==='/home');else{await n.p.waitForSelector('[role=alert]');assert(n.p.url().endsWith('/profile'))}
  assert.equal(n.writes.length,1);assert.equal(n.writes[0].method,'POST');assert.equal(n.writes[0].body[0].school,'Unlisted QA Campus');assert.equal(n.writes[0].body[0].credits,100);await n.p.close()
 }
 const a=await setup();await a.p.goto(base+'/auth');await a.p.waitForFunction(()=>location.pathname==='/home');await a.p.close()
 const c=await setup({chapterFailure:true});await c.p.goto(base+'/practice');await c.p.waitForSelector('select[aria-busy="true"]');assert.match(await c.p.$eval('select',e=>e.textContent),/Loading chapters/);await c.p.waitForSelector('[role=alert]');await c.p.screenshot({path:'/downloads/evater-chapter-retry.png',fullPage:true});await c.p.evaluate(()=>Array.from(document.querySelectorAll('button')).find(b=>b.textContent.includes('Retry chapters')).click());await c.p.waitForSelector('select[aria-busy="false"]');assert.equal(c.chapterCalls,2);await c.p.evaluate(()=>Array.from(document.querySelectorAll('button')).find(b=>b.textContent.includes('Mathematics')).click());await c.p.waitForFunction(()=>document.querySelector('select')?.textContent?.includes('Mathematics chapter'));assert.equal(c.chapterCalls,3);await c.p.evaluate(()=>Array.from(document.querySelectorAll('button')).find(b=>b.textContent.includes('Science')&&b.textContent.includes('Class 8')).click());await c.p.waitForFunction(()=>document.querySelector('select')?.textContent?.includes('Science chapter'));assert.equal(c.chapterCalls,3);await c.p.close()
 console.log('PASS: desktop keyboard/canonical selection, mobile touch/inner scroll/Other/Save/no overflow, unselected-query guard, custom preservation, new profile POST100-credit unchanged, saved-only redirect, failure/null stay, signed-in /auth, chapter loading/retry and subject cache. All remote requests mocked. No live writes.')
}finally{await browser.close();server.kill()}
