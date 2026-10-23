const $=s=>document.querySelector(s);
async function jget(u){const r=await fetch(u);if(!r.ok)throw new Error(u+' '+r.status);return r.json()}
async function jpost(u,d){const r=await fetch(u,{method:'POST',headers:{'Content-Type':'application/json'},
 body:JSON.stringify(d)});return r.json()}
let items=[],idx=0;
function progress(){const done=items.filter(i=>i.status!=='todo').length;
 $('#prog').textContent=done+' of '+items.length+' done or skipped';$('#bar div').style.width=(100*done/items.length)+'%'}
async function start(task,show){items=await jget('/api/'+task);const first=items.findIndex(i=>i.status==='todo');
 idx=Math.max(0,parseInt(location.hash.slice(1))||(first<0?0:first));await show();progress()}
function go(task,show,d){idx=Math.min(items.length-1,Math.max(0,idx+d));location.hash=idx;show()}
