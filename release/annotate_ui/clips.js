const FIELDS=[['n_objects','Road users in the clip (person, bicycle, car, motorcycle, bus, truck)'],
 ['missed_objects','Missed objects (never boxed while visible)'],['false_tracks','False tracks (a box on nothing, a reflection, a poster)'],
 ['class_confusions','Class confusions (car/truck/bus, rider/bicycle)'],['id_switches','Identity switches (one object, a new id)'],
 ['fragmentations','Fragmentations (an object loses its track and gets a new id later)']];
const COL=['#e6194b','#3cb44b','#4363d8','#f58231','#911eb4','#008080','#bcf60c','#fabebe'];
let meta=null,vd=$('#vd'),cv=$('#cv'),ctx=cv.getContext('2d'),end=0,vals={};
$('#counters').innerHTML=FIELDS.map(([k,t])=>'<label>'+t+'</label><button onclick="bump(\''+k+'\',-1)">&minus;</button> <b id="v_'+k+'">0</b> <button onclick="bump(\''+k+'\',1)">+</button>').join('');
function bump(k,d){vals[k]=Math.max(0,(vals[k]||0)+d);$('#v_'+k).textContent=vals[k]}
function frameNo(){return Math.round((vd.currentTime-meta.segment_start)*meta.fps)+1}
function draw(){requestAnimationFrame(draw);if(!meta)return;cv.width=vd.clientWidth;cv.height=vd.clientHeight;ctx.clearRect(0,0,cv.width,cv.height);
 const k=frameNo();$('#fr').textContent='frame '+k+' ('+(k-meta.k)+' of '+(meta.k1-meta.k)+')';
 (meta.tracks&&meta.tracks[k]||[]).forEach(([id,c,x,y,w,h])=>{const X=(x-w/2)*cv.width,Y=(y-h/2)*cv.height;
  ctx.strokeStyle=COL[((id%8)+8)%8];ctx.lineWidth=2;ctx.strokeRect(X,Y,w*cv.width,h*cv.height);ctx.fillStyle=ctx.strokeStyle;
  ctx.font='12px sans-serif';ctx.fillText(id+':'+c,X+2,Math.max(10,Y-3))});
 if(vd.currentTime>=end){vd.currentTime=meta.clip_start}}
function toggle(){vd.paused?vd.play():vd.pause()}function rate(r){vd.playbackRate=r}
function step(n){vd.pause();vd.currentTime=Math.min(end,Math.max(meta.clip_start,vd.currentTime+n/meta.fps))}
function restart(){vd.currentTime=meta.clip_start;vd.play()}
async function show(){const it=items[idx];$('#pos').textContent=(idx+1)+' / '+items.length+' ('+it.status+')';
 $('#msg').textContent='';meta=await jget('/api/clips/'+encodeURIComponent(it.id));vals=Object.assign({},meta.saved||{});
 FIELDS.forEach(([k])=>{vals[k]=parseInt(vals[k])||0;$('#v_'+k).textContent=vals[k]});$('#nt').value=vals.notes||'';
 if(!meta.available){vd.removeAttribute('src');meta=null;
  $('#msg').innerHTML='Video '+it.id+' is not on this machine. <button onclick="dl()">Download it with yt-dlp</button> or skip.';return}
 if(!meta.tracks){$('#msg').textContent='No detection file found for this segment: the tracks cannot be shown. Skip it.'}
 end=meta.clip_start+10;vd.src='/video/'+meta.row.video_id;vd.onloadedmetadata=()=>{vd.currentTime=meta.clip_start;vd.play()}}
async function dl(){$('#msg').textContent='Downloading...';const r=await jpost('/api/download/'+items[idx].row.video_id,{});
 if(r.ok)show();else $('#msg').textContent='Download failed: '+r.error}
async function save(status){const it=items[idx];const rec={status,notes:$('#nt').value};FIELDS.forEach(([k])=>rec[k]=vals[k]||0);
 await jpost('/api/clips/'+encodeURIComponent(it.id),rec);it.status=status;progress();go('clips',show,1)}
document.addEventListener('keydown',e=>{if(/INPUT|TEXTAREA/.test(e.target.tagName))return;
 if(e.key===' '){e.preventDefault();toggle()}else if(e.key==='Enter')save('done');
 else if(e.key==='ArrowRight')step(1);else if(e.key==='ArrowLeft')step(-1)});
requestAnimationFrame(draw);start('clips',show);
