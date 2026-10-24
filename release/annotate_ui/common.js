const $=s=>document.querySelector(s);
async function jget(u){const r=await fetch(u);if(!r.ok)throw new Error(u+' '+r.status);return r.json()}
async function jpost(u,d){const r=await fetch(u,{method:'POST',headers:{'Content-Type':'application/json'},
 body:JSON.stringify(d)});return r.json()}
let items=[],idx=0,currentShow=null;
function counts(){const c={todo:0,done:0,skipped:0};items.forEach(i=>{c[i.status]=(c[i.status]||0)+1});return c}
function progress(){const c=counts();
 $('#prog').textContent=c.done+' done, '+c.skipped+' skipped, '+c.todo+' to do (of '+items.length+')';
 $('#bar div').style.width=(100*(c.done+c.skipped)/items.length)+'%'}
function say(text,bad){const t=$('#toast');t.textContent=text;t.style.color=bad?'#b00020':'#1a7f37'}
async function start(task,show){items=await jget('/api/'+task);currentShow=show;
 const want=new URLSearchParams(location.search).get('show');  // ?show=skipped opens at the first skipped item
 const first=items.findIndex(i=>i.status===(want||'todo'));
 idx=Math.max(0,parseInt(location.hash.slice(1))||(first<0?0:first));await show();progress()}
function go(task,show,d){const n=Math.min(items.length-1,Math.max(0,idx+d));
 if(n===idx&&d>0){const c=counts();say('This is the last item.'+(c.todo+c.skipped
  ?' Still open: '+c.todo+' to do, '+c.skipped+' skipped. Press J for the next to do, K for the next skipped.'
  :' All items are done.'),false)}
 else if(n===idx&&d<0)say('This is the first item.',true);else say('',false);
 idx=n;location.hash=idx;show()}
function jump(status){const n=items.length;
 for(let k=1;k<=n;k++){const i=(idx+k)%n;if(items[i].status===status){idx=i;location.hash=idx;say('',false);currentShow();return}}
 say('No '+(status==='todo'?'unfinished':status)+' items left.',false)}
document.addEventListener('keydown',e=>{if(/INPUT|TEXTAREA|SELECT|VIDEO/.test(e.target.tagName)||e.ctrlKey||e.metaKey)return;
 if(e.key==='j')jump('todo');else if(e.key==='k')jump('skipped')});
