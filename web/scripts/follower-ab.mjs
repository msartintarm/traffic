import { chromium } from "playwright";
const browser = await chromium.connectOverCDP(process.argv[2] ?? "http://localhost:9222");
const ctx = browser.contexts()[0];
const page = ctx.pages().find(p=>p.url().includes("scenario")) ?? (await ctx.newPage());
const url = "http://localhost:3001/?scenario=sf&compute=threads&debug=1&warmup=1";
await page.goto(url, {waitUntil:"domcontentloaded"});
await page.waitForTimeout(6000);
await page.waitForFunction(()=>!!window.__stats,null,{timeout:45000});
let ew=null; for(let i=0;i<30&&!ew;i++){ew=page.workers().find(w=>/engineWorker/.test(w.url())&&!w.url().startsWith("blob:")); if(!ew)await page.waitForTimeout(1000);}
await page.waitForFunction(()=>window.__stats&&!window.__stats.warmup,null,{timeout:180000,polling:2000}).catch(()=>{});
await page.evaluate(()=>window.__ctl&&window.__ctl({type:"pause"}));
const one=(on)=>ew.evaluate((on)=>{const s=globalThis.__sim; s.set_follower_lod(on); s.debug_bench_steps(20); const n=s.vehicle_count(); const ms=s.debug_bench_steps(60); return ms/60/Math.max(1,n)*1000;},on);
const med=a=>a.slice().sort((x,y)=>x-y)[a.length>>1];
const off=[],onn=[]; for(let i=0;i<12;i++){off.push(await one(false)); onn.push(await one(true));}
const mo=med(off),mn=med(onn);
console.log(`[FAB] follower OFF ${mo.toFixed(4)} us/car | ON ${mn.toFixed(4)} us/car | delta ${(((mo-mn)/mo)*100).toFixed(1)}%`);
process.exit(0);
