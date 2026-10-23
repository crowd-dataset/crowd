const COL=['#e6194b','#3cb44b','#4363d8','#f58231','#911eb4','#008080'];
const VCOL={correct:'#2da44e',false_positive:'#d1242f',wrong_class:'#f58231'};
let classes=[],preds=[],missed=[],sel=null,cur=0,meta=null,drag=null,img=$('#img'),cv=$('#cv'),ctx=cv.getContext('2d');
function chips(){$('#chips').innerHTML=classes.map((c,i)=>'<span style="background:'+COL[i]+';'+(i===cur?'outline:3px solid #000':'')+
 '" onclick="pick('+i+')">'+(i+1)+' '+c+'</span>').join('');$('#cls').textContent=classes[cur]}
function pick(n){if(sel&&sel.kind==='pred'){const b=preds[sel.i];if(classes[n]===b.c){b.v='correct';b.t=''}else{b.v='wrong_class';b.t=classes[n]}}
 else if(sel&&sel.kind==='missed'){missed[sel.i].c=classes[n]}else cur=n;chips();draw()}
function setV(v){if(sel&&sel.kind==='pred'){preds[sel.i].v=v;preds[sel.i].t='';draw()}}
function delSel(){if(!sel)return;if(sel.kind==='missed'){missed.splice(sel.i,1)}else{preds[sel.i].v='correct';preds[sel.i].t=''}sel=null;draw()}
function undo(){missed.pop();sel=null;draw()}
function rect(b){return[(b.x-b.w/2)*cv.width,(b.y-b.h/2)*cv.height,b.w*cv.width,b.h*cv.height]}
function draw(){cv.width=img.clientWidth;cv.height=img.clientHeight;ctx.clearRect(0,0,cv.width,cv.height);ctx.font='12px sans-serif';
 preds.forEach((b,i)=>{const[x,y,w,h]=rect(b);const isSel=sel&&sel.kind==='pred'&&sel.i===i;ctx.strokeStyle=VCOL[b.v];ctx.lineWidth=isSel?5:2;
  ctx.setLineDash(b.v==='false_positive'?[6,4]:[]);ctx.strokeRect(x,y,w,h);ctx.setLineDash([]);ctx.fillStyle=VCOL[b.v];
  const label=b.v==='false_positive'?'X '+b.c:b.v==='wrong_class'?b.c+' → '+b.t:b.c+' '+b.conf.toFixed(2);ctx.fillText(label,x+2,Math.max(10,y-3))});
 missed.forEach((b,i)=>{const[x,y,w,h]=rect(b);const isSel=sel&&sel.kind==='missed'&&sel.i===i;ctx.strokeStyle='#1f6feb';ctx.lineWidth=isSel?5:2;
  ctx.setLineDash([2,3]);ctx.strokeRect(x,y,w,h);ctx.setLineDash([]);ctx.fillStyle='#1f6feb';ctx.fillText('missed '+b.c,x+2,Math.max(10,y-3))});
 if(drag){ctx.strokeStyle=COL[cur];ctx.setLineDash([4,3]);ctx.lineWidth=2;ctx.strokeRect(drag.x0,drag.y0,drag.x1-drag.x0,drag.y1-drag.y0);ctx.setLineDash([])}
 $('#n_c').textContent=preds.filter(b=>b.v==='correct').length;$('#n_f').textContent=preds.filter(b=>b.v==='false_positive').length;
 $('#n_w').textContent=preds.filter(b=>b.v==='wrong_class').length;$('#n_m').textContent=missed.length}
function pos(e){const r=cv.getBoundingClientRect();return[e.clientX-r.left,e.clientY-r.top]}
cv.onmousedown=e=>{const[x,y]=pos(e);drag={x0:x,y0:y,x1:x,y1:y}};
cv.onmousemove=e=>{if(drag){[drag.x1,drag.y1]=pos(e);draw()}};
cv.onmouseup=e=>{if(!drag)return;const{x0,y0,x1,y1}=drag;drag=null;const w=Math.abs(x1-x0),h=Math.abs(y1-y0);
 if(w<4||h<4){const inside=b=>Math.abs(x0/cv.width-b.x)<b.w/2&&Math.abs(y0/cv.height-b.y)<b.h/2;
  const area=b=>b.w*b.h;let hit=null;
  preds.forEach((b,i)=>{if(inside(b)&&(!hit||area(b)<hit.a))hit={kind:'pred',i,a:area(b)}});
  missed.forEach((b,i)=>{if(inside(b)&&(!hit||area(b)<hit.a))hit={kind:'missed',i,a:area(b)}});
  sel=hit?{kind:hit.kind,i:hit.i}:null}
 else{missed.push({c:classes[cur],x:(x0+x1)/2/cv.width,y:(y0+y1)/2/cv.height,w:w/cv.width,h:h/cv.height});sel=null}draw()};
async function show(){const it=items[idx];$('#pos').textContent=(idx+1)+' / '+items.length+' ('+it.status+')';$('#msg').textContent='';
 meta=await jget('/api/review/'+encodeURIComponent(it.id));classes=meta.classes;chips();sel=null;
 const saved=meta.saved||{};preds=(saved.preds)||meta.preds||[];missed=saved.missed||[];
 if(!meta.available){img.removeAttribute('src');cv.width=cv.height=0;preds=[];missed=[];
  $('#msg').innerHTML='Video '+meta.row.video_id+' is not on this machine. <button onclick="dl()">Download it with yt-dlp</button> or skip.';return}
 const fr=await fetch('/frame/'+encodeURIComponent(it.id)+'.jpg');
 if(!fr.ok){img.removeAttribute('src');cv.width=cv.height=0;preds=[];missed=[];
  $('#msg').innerHTML='<b style="color:#b00020">This frame cannot be shown.</b> '+(await fr.text())+' <br>Skip it (S).';return}
 if(!meta.has_detections)$('#msg').innerHTML='<b style="color:#b00020">No detection file was found for this segment, so there are no boxes to check.</b> Skip it (S), or add the folder with --bbox.';
 img.onload=draw;img.src=URL.createObjectURL(await fr.blob())}
async function dl(){$('#msg').textContent='Downloading...';const r=await jpost('/api/download/'+items[idx].row.video_id,{});
 if(r.ok)show();else $('#msg').textContent='Download failed: '+r.error}
async function save(status){if(status==='done'&&meta&&!meta.has_detections){alert('There is no detection file for this segment: skip it.');return}
 const it=items[idx];await jpost('/api/review/'+encodeURIComponent(it.id),{status,preds,missed,k:meta&&meta.k});
 it.status=status;progress();go('review',show,1)}
window.onresize=draw;
document.addEventListener('keydown',e=>{if(/INPUT|TEXTAREA/.test(e.target.tagName))return;
 if(e.key>='1'&&e.key<='6')pick(+e.key-1);else if(e.key==='Enter')save('done');else if(e.key==='s')save('skipped');
 else if(e.key==='x')setV('false_positive');else if(e.key==='c')setV('correct');else if(e.key==='z')undo();
 else if(e.key==='Delete'||e.key==='Backspace')delSel();else if(e.key==='ArrowRight')go('review',show,1);
 else if(e.key==='ArrowLeft')go('review',show,-1)});
start('review',show);
