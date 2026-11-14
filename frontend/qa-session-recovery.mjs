// Real app/router/Supabase SDK, all backend requests intercepted. No real accounts.
import puppeteer from 'puppeteer'
import { createServer } from 'node:http'
import { readFile, writeFile, mkdir } from 'node:fs/promises'
import { join, extname } from 'node:path'
const output = process.env.QA_EVIDENCE || '/downloads/evater-session-evidence'
await mkdir(output, { recursive: true })
const server = createServer(async (req,res) => {
  let path = new URL(req.url,'http://local').pathname, data
  try { data = await readFile(join('dist',path)) } catch { path='/index.html'; data=await readFile('dist/index.html') }
  res.setHeader('Content-Type', ({'.js':'application/javascript','.css':'text/css','.png':'image/png','.html':'text/html','.svg':'image/svg+xml'})[extname(path)] || 'application/octet-stream')
  res.end(data)
})
await new Promise(r=>server.listen(0,'127.0.0.1',r))
const origin=`http://127.0.0.1:${server.address().port}`
const browser=await puppeteer.launch({executablePath:'/usr/bin/google-chrome',headless:true,args:['--no-sandbox']})
const results=[], unexpected=[], pageErrors=[]
const user={id:'00000000-0000-4000-8000-000000000099',email:'fixture@example.invalid',aud:'authenticated',role:'authenticated',created_at:'2026-11-01T00:00:00Z',app_metadata:{provider:'email'},user_metadata:{}}
const session={access_token:'fixture-not-a-real-token',refresh_token:'fixture-not-a-real-refresh-token',expires_at:Math.floor(Date.now()/1000)+3600,expires_in:3600,token_type:'bearer',user}
const key='sb-qfxrfwgipsygoozlnkwb-auth-token'
const assert=(x,m)=>{if(!x)throw Error(m)}
const click=async(page,text)=>{const h=await page.evaluateHandle(t=>[...document.querySelectorAll('button')].find(b=>b.textContent.includes(t)),text);assert(h.asElement(),'No button '+text);await h.asElement().click()}
try {
for (const width of [320,390,1280]) for (const scenario of ['valid','revoked','unavailable','signout-missing','signout-failure','normal-signout','delayed-signout']) {
 const context=await browser.createBrowserContext(),page=await context.newPage()
 await page.setViewport({width,height:900})
 page.setDefaultTimeout(4000)
 let signout=false, requestCount=0
 page.on('pageerror',e=>pageErrors.push(e.message))
 await page.evaluateOnNewDocument((k,s)=>localStorage.setItem(k,JSON.stringify(s)),key,session)
 await page.setRequestInterception(true)
 page.on('request',async r=>{
  const u=new URL(r.url()); if(u.origin===origin)return r.continue()
  if(['data:','blob:'].includes(u.protocol))return r.continue()
  const headers={'access-control-allow-origin':'*','access-control-allow-headers':'*','access-control-allow-methods':'GET,POST,OPTIONS','x-supabase-api-version':'2024-01-01','access-control-expose-headers':'x-supabase-api-version'}
  const respond=(status,body)=>r.respond({status,headers,contentType:'application/json',body:JSON.stringify(body)})
  if(u.hostname.endsWith('.supabase.co')) {
   requestCount++
   if(r.method()==='OPTIONS')return r.respond({status:204,headers})
   if(u.pathname==='/auth/v1/user') {
    if(scenario==='delayed-signout' && signout)await new Promise(r=>setTimeout(r,150))
    if(scenario==='revoked'||(scenario==='signout-missing'&&signout))return respond(403,{code:'session_not_found',msg:'Session from session_id claim in JWT does not exist'})
    if(scenario==='unavailable')return respond(502,{msg:'Fixture upstream unavailable'})
    return respond(200,user)
   }
   if(u.pathname==='/auth/v1/logout'){
    signout=true
    if(scenario==='signout-missing')return respond(403,{code:'session_not_found',msg:'Session from session_id claim in JWT does not exist'})
    if(scenario==='signout-failure')return respond(400,{code:'unexpected_failure',msg:'Fixture logout failure'})
    return r.respond({status:204,headers})
   }
   if(u.pathname==='/rest/v1/Users')return respond(200,{id:'fixture-profile',created_by:user.id,name:'Fixture Learner',user_name:'Fixture Learner',email:user.email,grade:8,credits:100,school:'Fixture School'})
   if(u.pathname.startsWith('/rest/v1/'))return respond(200,[])
  }
  if(u.hostname.endsWith('.modal.run'))return respond(200,[])
  if(['fonts.googleapis.com','fonts.gstatic.com'].includes(u.hostname))return r.abort()
  unexpected.push({scenario,host:u.hostname,path:u.pathname}); return r.abort()
 })
 try {
  const destination = scenario==='revoked' ? '/practice?subject=mathematics' : '/home'
  await page.goto(origin+destination,{waitUntil:'networkidle0'})
  if(scenario==='revoked'){
   await page.waitForSelector('input[type=email]')
   assert(new URL(page.url()).pathname==='/auth','Revoked session not on auth')
   assert(await page.evaluate(k=>!localStorage.getItem(k),key),'SDK did not remove revoked session')
   assert(new URL(page.url()).searchParams.get('next')===destination,'Lost destination')
  } else if(scenario==='unavailable') {
   await page.waitForFunction(()=>document.body.innerText.includes('Unable to verify'))
   assert(await page.evaluate(k=>!!localStorage.getItem(k),key),'Transient failure removed session')
   assert(!await page.evaluate(()=>document.body.innerText.includes('Good to see you')),'Cached workspace leaked through')
  } else {
   await page.waitForFunction(()=>document.body.innerText.includes('Good to see you, Fixture'))
   if(scenario!=='valid'){
    if(width<768)await page.click('button[aria-label="Open navigation menu"]')
    else await click(page,'Fixture Learner')
    await click(page,'Sign out')
    if(scenario==='signout-failure'){
     await page.waitForSelector('[role=alert]')
     assert(await page.$eval('[role=alert]',e=>e.textContent.includes('Could not sign out')),'Missing visible failure')
     assert(new URL(page.url()).pathname==='/home','Failure navigated away')
     assert(await page.evaluate(k=>!!localStorage.getItem(k),key),'Failure discarded session')
    }else{
     await page.waitForSelector('input[type=email]')
     assert(new URL(page.url()).pathname==='/auth','Sign out bounced home')
     assert(await page.evaluate(k=>!localStorage.getItem(k),key),'Sign out retained session')
    }
   }
  }
  assert(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth),'Horizontal overflow')
  assert(requestCount<25,'Unexpected auth loop')
  if(['revoked','unavailable','signout-failure'].includes(scenario))await page.screenshot({path:join(output,`${width}-${scenario}.png`),fullPage:true})
  results.push({width,scenario,status:'PASS',requestCount});console.log(width,scenario,'PASS')
 }catch(e){results.push({width,scenario,status:'FAIL',reason:e.message});console.log(width,scenario,'FAIL',e.message);await page.screenshot({path:join(output,`${width}-${scenario}-FAIL.png`),fullPage:true})}
 await context.close()
}
}finally{await browser.close();server.close();await writeFile(join(output,'results.json'),JSON.stringify({results,unexpected,pageErrors},null,2));console.log(JSON.stringify({results,unexpected,pageErrors},null,2))}
if(results.some(r=>r.status!=='PASS')||unexpected.length||pageErrors.length)process.exitCode=1
