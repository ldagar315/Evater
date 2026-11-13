// Loopback-only component regression: no account creation, email or OAuth request.
import { build } from 'esbuild'
import puppeteer from 'puppeteer'
import { createServer } from 'node:http'
import { mkdtemp, writeFile, readFile, mkdir, rm } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { resolve, join } from 'node:path'
import { execFileSync } from 'node:child_process'

const cwd = process.cwd()
const baseline = process.argv.includes('--baseline')
const evidenceIndex = process.argv.indexOf('--evidence-dir')
const evidenceDir = evidenceIndex < 0 ? await mkdtemp(join(tmpdir(), 'evater-auth-evidence-')) : resolve(process.argv[evidenceIndex + 1])
await mkdir(evidenceDir, { recursive: true })
const temp = await mkdtemp(join(cwd, '.qa-auth-'))
const componentPath = resolve(cwd, 'src/components/auth/AuthForm.tsx')
const original = baseline ? execFileSync('git', ['show', 'f7fe706:frontend/src/components/auth/AuthForm.tsx'], { encoding: 'utf8' }) : null
const mock = `
export const supabaseConfigured = true;
const response = async (method) => {
 window.calls.push(method);
 const scenario = window.scenario;
 if(scenario === 'throw') throw new Error('Fixture failure');
 if(scenario === 'pending') return new Promise(resolve => {window.releaseAuth = () => resolve({data:{session:null}, error:null})});
 if(scenario === 'error') return {data:null,error:{message:'Fixture service error'}};
 if(scenario === 'credentials') return {data:null,error:{message:'Invalid login credentials'}};
 if(method === 'google') return {data:{provider:'google',url:'https://accounts.example.invalid/fixture'},error:null};
 return {data: {session:scenario === 'session' ? {user:{id:'fixture-user'}} : null}, error:null};
};
export const useAuthContext = () => ({user:null,error:null,signUp:()=>response('signup'),signIn:()=>response('signin'),signInWithGoogle:()=>response('google')});
`
await writeFile(join(temp, 'mock.ts'), mock)
const result = await build({
 stdin: { contents: `import React from 'react'; import {createRoot} from 'react-dom/client'; import {MemoryRouter} from 'react-router-dom'; import {AuthForm} from './src/components/auth/AuthForm'; window.calls=[]; window.successes=0; window.scenario=new URLSearchParams(location.search).get('case') || 'confirmation'; createRoot(document.getElementById('root')).render(<MemoryRouter initialEntries={[new URLSearchParams(location.search).get('mode') === 'signin' ? '/auth' : '/auth?mode=signup']}><AuthForm onSuccess={()=>{window.successes++; document.body.dataset.successes=String(window.successes)}}/></MemoryRouter>);`, loader:'tsx',resolveDir:cwd },
 bundle:true,write:false,format:'esm',jsx:'automatic',define:{'process.env.NODE_ENV':'"production"'},
 plugins:[{name:'fixture-only-auth',setup(build){
  build.onResolve({filter: /contexts\/AuthContext$|lib\/supabaseConfig$/},()=>({path:join(temp,'mock.ts')}))
  if(baseline)build.onLoad({filter:/components\/auth\/AuthForm\.tsx$/},()=>({contents:original,loader:'tsx',resolveDir:resolve(cwd,'src/components/auth')}))
 }}],
})
await writeFile(join(temp,'bundle.js'),result.outputFiles[0].text)
execFileSync(resolve(cwd,'node_modules/.bin/tailwindcss'),['-i','src/index.css','-o',join(temp,'style.css'),'--minify'],{cwd,stdio:'pipe'})
// Use the app CSS without its network font import: this fixture is strictly offline.
await writeFile(join(temp,'style.css'),(await readFile(join(temp,'style.css'),'utf8')).replace(/@import\s+url\([^)]*\);/g,''))
const server = createServer(async(req,res)=>{
 const path = req.url.split('?')[0]
 if(path === '/bundle.js' || path === '/style.css') {
  res.setHeader('Content-Type',path.endsWith('js')?'application/javascript':'text/css');res.end(await readFile(join(temp,path.slice(1))));return
 }
 if(path==='/Evater_logo_2.png'){res.setHeader('Content-Type','image/png');res.end(await readFile(join(cwd,'public/Evater_logo_2.png')));return}
 res.setHeader('Content-Type','text/html');res.end('<!doctype html><meta name="viewport" content="width=device-width,initial-scale=1"><link rel="stylesheet" href="/style.css"><div id="root"></div><script type="module" src="/bundle.js"></script>')
})
await new Promise(resolve=>server.listen(0,'127.0.0.1',resolve))
const origin=`http://127.0.0.1:${server.address().port}`
const browser = await puppeteer.launch({executablePath:process.env.CHROME_PATH || '/usr/bin/google-chrome',headless:true,args:['--no-sandbox','--disable-setuid-sandbox']})
const records=[];const failures=[];const externalRequests=[];const pageErrors=[]
const check=(ok,message)=>{if(!ok)throw new Error(message)}
try {
 for(const width of [320,390,1280]) {
  const page=await browser.newPage();await page.setViewport({width,height:900})
  await page.setRequestInterception(true)
  page.on('request',r=>{if(r.url().startsWith(origin))r.continue();else{externalRequests.push(r.url());r.abort()}})
  page.on('pageerror',e=>pageErrors.push(e.message))
  const open=async(scenario,mode='signup')=>{await page.goto(`${origin}/?case=${scenario}&mode=${mode}`,{waitUntil:'networkidle0'});await page.waitForSelector('form')}
  const button=async(text)=>{const found=await page.evaluateHandle(t=>[...document.querySelectorAll('button')].find(x=>x.textContent.includes(t)),text);check(found.asElement(),'button missing:'+text);await found.asElement().click()}
  const fill=async(mismatch=false)=>{await page.type('input[type=email]','student@example.invalid');const fields=await page.$$('input[type=password]');for(let i=0;i<fields.length;i++)await fields[i].type(mismatch&&i===1?'fixture-other':'fixture-password')}
  const submit=async()=>{await fill();await button('Create Your Account');await page.waitForFunction(()=>window.calls.length>0)}
  const settled=async()=>{await page.waitForFunction(()=>!document.querySelector('button[type=submit] svg.animate-spin'))}
  const snap=async(name)=>{await page.screenshot({path:join(evidenceDir,`${baseline?'baseline-':''}${width}-${name}.png`),fullPage:true})}
  const test=async(name,run)=>{try{await run();records.push({width,name,status:'PASS'})}catch(e){records.push({width,name,status:'FAIL',reason:e.message});failures.push(`${width}/${name}: ${e.message}`)}}
  await test('signup-without-session-stays-on-form',async()=>{
   await open('confirmation');await submit();await settled();
   if(baseline){check(await page.evaluate(()=>window.successes===1),'baseline defect did not reproduce');await snap('false-success');return}
   await page.waitForSelector('[role=status]');check(await page.evaluate(()=>window.successes===0),'premature completion callback');
   check((await page.$$('form')).length===0,'confirmation left resubmittable signup form');
   check((await page.$$('input[type=password]')).length===0,'confirmation retains password fields');await snap('confirmation')
  })
  await test('oauth-initiation-is-not-authentication',async()=>{
   await open('confirmation','signin');await button('Continue with Google');await settled();
   check(await page.evaluate(expected=>window.successes===expected,baseline?1:0),'OAuth initiation called completion');await snap('oauth')
  })
  if(!baseline){
   await test('confirmation-to-signin-no-extra-request',async()=>{await open('confirmation');await submit();await page.waitForSelector('[role=status]');await button('Back to sign in');await page.waitForSelector('form');check((await page.$$('input[type=password]')).length===1,'not signin form');check(await page.evaluate(()=>window.calls.length===1),'switch made extra request');check(await page.$eval('input[type=email]',e=>e.value==='student@example.invalid'),'email lost');check(await page.$eval('input[type=password]',e=>e.value===''),'password retained');await snap('back-to-signin')})
   for(const mode of ['signup','signin'])await test(`${mode}-with-session-completes-once`,async()=>{await open('session',mode);await fill();await button(mode==='signup'?'Create Your Account':'Sign In to Evater');await page.waitForFunction(()=>window.successes===1);check(await page.evaluate(()=>window.calls.length===1),'duplicate request')})
   await test('signin-without-session-fails-visible',async()=>{await open('confirmation','signin');await fill();await button('Sign In to Evater');await page.waitForFunction(()=>document.body.innerText.includes('Sign-in did not return a session'));check(await page.evaluate(()=>window.successes===0),'null signin completes');check(!(await page.$('[role=status]')),'signin asks email confirmation');await snap('missing-session')})
   for(const mode of ['signup','signin'])for(const scenario of ['error','throw'])await test(`${mode}-${scenario}-preserved`,async()=>{await open(scenario,mode);await fill();await button(mode==='signup'?'Create Your Account':'Sign In to Evater');await page.waitForFunction(()=>document.body.innerText.includes('Fixture service error')||document.body.innerText.includes('An unexpected error occurred'));check(await page.evaluate(()=>window.successes===0),'error completed');check(!(await page.$('[role=status]')),'error confirmation')})
   await test('password-mismatch-no-request',async()=>{await open('confirmation');await fill(true);await button('Create Your Account');await page.waitForFunction(()=>document.body.innerText.includes('Passwords do not match'));check(await page.evaluate(()=>window.calls.length===0&&window.successes===0),'mismatch submitted')})
   for(const scenario of ['error','throw'])await test(`oauth-${scenario}-preserved`,async()=>{await open(scenario);await button('Continue with Google');await page.waitForFunction(()=>document.body.innerText.includes('Fixture service error')||document.body.innerText.includes('Failed to sign in with Google'));check(await page.evaluate(()=>window.successes===0),'oautherror completed')})
   await test('pending-request-no-success',async()=>{await open('pending');await submit();check(await page.evaluate(()=>window.successes===0),'pending completed');check(await page.$eval('button[type=submit]',b=>b.disabled),'submit enabled while pending');await page.evaluate(()=>window.releaseAuth());await page.waitForSelector('[role=status]');check(await page.evaluate(()=>window.calls.length===1&&window.successes===0),'pending duplicated')})
   await test('credentials-guidance-preserved',async()=>{await open('credentials','signin');await fill();await button('Sign In to Evater');await page.waitForFunction(()=>document.body.innerText.includes('New here?'));check(await page.evaluate(()=>window.successes===0),'credentials completed')})
   await test('no-horizontal-overflow',async()=>{await open('confirmation');await submit();await page.waitForSelector('[role=status]');check(await page.evaluate(()=>document.documentElement.scrollWidth<=window.innerWidth),'horizontal overflow')})
  }
  await page.close()
 }
 check(externalRequests.length===0,'unexpected external requests');check(pageErrors.length===0,'browser errors')
}catch(error){
 failures.push('suite: '+error.message)
}finally{
 await browser.close();await new Promise(resolve=>server.close(resolve));await rm(temp,{recursive:true,force:true})
 const report={baseline,records,externalRequests,pageErrors,failures};await writeFile(join(evidenceDir,baseline?'baseline-results.json':'results.json'),JSON.stringify(report,null,2));console.log(JSON.stringify(report,null,2));console.log('Evidence:',evidenceDir)
}
if(failures.length)process.exitCode=1
