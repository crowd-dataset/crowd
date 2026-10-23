const COL=['#e6194b','#3cb44b','#4363d8','#f58231','#911eb4','#008080'];
let classes=[],boxes=[],sel=-1,cur=0,mode='blind',drag=null,meta=null,img=$('#img'),cv=$('#cv'),ctx=cv.getContext('2d');
function chips(){$('#chips').innerHTML=classes.map((c,i)=>'<span style="background:'+COL[i]+';'+(i===cur?'outline:3px solid #000':'')+
 '" onclick="cur='+i+';chips()">'+(i+1)+' '+c+'</span>').join('');$('#cls').textContent=classes[cur]}
function draw(){cv.width=img.clientWidth;cv.height=img.clientHeight;ctx.clearRect(0,0,cv.width,cv.height);ctx.lineWidth=2;
 boxes.forEach((b,i)=>{const x=(b.x-b.w/2)*cv.width,y=(b.y-b.h/2)*cv.height,w=b.w*cv.width,h=b.h*cv.height,k=classes.indexOf(b.c);
  ctx.strokeStyle=COL[k];ctx.lineWidth=i===sel?4:2;ctx.strokeRect(x,y,w,h);ctx.fillStyle=COL[k];ctx.font='12px sans-serif';
  ctx.fillText(b.c,x+2,Math.max(10,y-3))});
 if(drag){ctx.strokeStyle=COL[cur];ctx.setLineDash([4,3]);ctx.strokeRect(drag.x0,drag.y0,drag.x1-drag.x0,drag.y1-drag.y0);ctx.setLineDash([])}
 $('#nb').textContent=boxes.length}
function pos(e){const r=cv.getBoundingClientRect();return[e.clientX-r.left,e.clientY-r.top]}
cv.onmousedown=e=>{const[x,y]=pos(e);drag={x0:x,y0:y,x1:x,y1:y}};
cv.onmousemove=e=>{if(drag){[drag.x1,drag.y1]=pos(e);draw()}};
cv.onmouseup=e=>{if(!drag)return;const{x0,y0,x1,y1}=drag;drag=null;const w=Math.abs(x1-x0),h=Math.abs(y1-y0);
 if(w<4||h<4){const hit=boxes.map((b,i)=>[i,b]).reverse().find(([i,b])=>Math.abs(x0/cv.width-b.x)<b.w/2&&Math.abs(y0/cv.height-b.y)<b.h/2);
  sel=hit?hit[0]:-1}
 else boxes.push({c:classes[cur],x:(x0+x1)/2/cv.width,y:(y0+y1)/2/cv.height,w:w/cv.width,h:h/cv.height});draw()};
function undo(){boxes.pop();sel=-1;draw()}function delSel(){if(sel>=0){boxes.splice(sel,1);sel=-1;draw()}}
async function prelabel(){const y=await jget('/api/frames/'+encodeURIComponent(items[idx].id)+'/yolo');boxes=y;mode='prelabel';draw()}
async function show(){const it=items[idx];$('#pos').textContent=(idx+1)+' / '+items.length+' ('+it.status+')';
 $('#msg').textContent='';meta=await jget('/api/frames/'+encodeURIComponent(it.id));classes=meta.classes;chips();
 boxes=(meta.saved&&meta.saved.boxes)||[];mode=(meta.saved&&meta.saved.mode)||'blind';sel=-1;
 if(!meta.available){img.removeAttribute('src');cv.width=cv.height=0;
  $('#msg').innerHTML='Video '+meta.row.video_id+' is not on this machine. <button onclick="dl()">Download it with yt-dlp</button> or skip.';return}
 const fr=await fetch('/frame/'+encodeURIComponent(it.id)+'.jpg');
 if(!fr.ok){img.removeAttribute('src');cv.width=cv.height=0;boxes=[];draw();
  $('#msg').innerHTML='<b style="color:#b00020">This frame cannot be shown.</b> '+(await fr.text())+' <br>Skip it (S).';return}
 img.onload=draw;img.src=URL.createObjectURL(await fr.blob())}
async function dl(){$('#msg').textContent='Downloading...';const r=await jpost('/api/download/'+items[idx].row.video_id,{});
 if(r.ok)show();else $('#msg').textContent='Download failed: '+r.error}
async function save(status,none){if(status==='done'&&!none&&!boxes.length&&!confirm('No boxes: save as a frame without objects?'))return;
 const it=items[idx];await jpost('/api/frames/'+encodeURIComponent(it.id),{status,boxes:none?[]:boxes,mode,k:meta&&meta.k});
 it.status=status;progress();go('frames',show,1)}
window.onresize=draw;
document.addEventListener('keydown',e=>{if(/INPUT|TEXTAREA/.test(e.target.tagName))return;
 if(e.key>='1'&&e.key<='6'){cur=+e.key-1;chips()}else if(e.key==='Enter')save('done');else if(e.key==='e')save('done',true);
 else if(e.key==='s')save('skipped');else if(e.key==='z')undo();else if(e.key==='p')prelabel();
 else if(e.key==='Delete'||e.key==='Backspace')delSel();else if(e.key==='ArrowRight')go('frames',show,1);
 else if(e.key==='ArrowLeft')go('frames',show,-1)});
start('frames',show);
