let meta=null,vd=$('#vd'),cv=$('#cv'),ctx=cv.getContext('2d'),zoomed=true,piece=0,centre=null;
const KEYS=['one','switch','false','unsure'];
function radios(){$('#q1').innerHTML=meta.ONE.map(([v,t],i)=>
 '<label class="opt"><input type="radio" name="q1" value="'+v+'" autocomplete="off"> '+(i+1)+'. '+t+'</label>').join('')}
function picked(){const e=document.querySelector('input[name=q1]:checked');return e?e.value:''}
function setPicked(v){document.querySelectorAll('input[name=q1]').forEach(e=>e.checked=e.value===v)}
const tOf=k=>meta.segment_start+(k-1)/meta.fps;
function frameNo(){return Math.round((vd.currentTime-meta.segment_start)*meta.fps)+1}
function startPiece(i){piece=i%meta.pieces.length;vd.currentTime=tOf(meta.pieces[piece][0]);centre=null}
function draw(){requestAnimationFrame(draw);if(!meta||!meta.pieces)return;const W=cv.width,H=cv.height;
 const k=frameNo(),p=meta.pieces[piece];
 if(k>p[1]){if(meta.pieces.length>1||!vd.paused)startPiece(piece+1)}
 const b=meta.boxes[k]||meta.boxes[k-1]||meta.boxes[k+1]||meta.boxes[Math.max(p[0],Math.min(p[1],k))],vw=vd.videoWidth,vh=vd.videoHeight;if(!vw)return;
 let sx=0,sy=0,sw=vw,sh=vh;
 if(zoomed&&b){const side=Math.min(Math.min(vw,vh),Math.max(b[2]*vw,b[3]*vh)*3.5+80),cx=b[0]*vw,cy=b[1]*vh;
  centre=centre?[centre[0]+(cx-centre[0])*0.25,centre[1]+(cy-centre[1])*0.25]:[cx,cy];  // smoothed: the picture does not jitter
  sw=sh=side;sx=Math.min(Math.max(centre[0]-side/2,0),vw-side);sy=Math.min(Math.max(centre[1]-side/2,0),vh-side)}
 const ar=sw/sh;cv.height=H;ctx.clearRect(0,0,W,H);ctx.fillStyle='#000';ctx.fillRect(0,0,W,H);
 const dw=ar>=1?W:W*ar,dh=ar>=1?W/ar:H,ox=(W-dw)/2,oy=(H-dh)/2;
 ctx.drawImage(vd,sx,sy,sw,sh,ox,oy,dw,dh);
 const exact=meta.boxes[k]||meta.boxes[k-1]||meta.boxes[k+1];  // video time is not exact to the frame
 if(exact){const x=(exact[0]-exact[2]/2)*vw,y=(exact[1]-exact[3]/2)*vh;
  ctx.strokeStyle='#e8212b';ctx.lineWidth=3;ctx.strokeRect(ox+(x-sx)/sw*dw,oy+(y-sy)/sh*dh,exact[2]*vw/sw*dw,exact[3]*vh/sh*dh)}
 $('#fr').textContent='piece '+(piece+1)+' of '+meta.pieces.length+', '+((k-p[0])/meta.fps).toFixed(1)+' s'+(exact?'':' (no box in this frame)')}
function toggle(){vd.paused?vd.play():vd.pause()}function rate(r){vd.playbackRate=r}
function restart(){startPiece(0);vd.play()}
async function show(){const it=items[idx];$('#pos').textContent=(idx+1)+' / '+items.length+' ('+it.status+')';$('#msg').textContent='';
 meta=await jget('/api/tracks/'+encodeURIComponent(it.id));$('#tid').textContent=meta.row.track_id;$('#cls').textContent=meta.row.class;
 $('#prev').textContent=(meta.saved||{}).previous?'To check again. Your earlier answer: '+meta.saved.previous.one_object:'';radios();setPicked((meta.saved||{}).one_object||'');$('#nt').value=(meta.saved||{}).notes||'';
 if(!meta.available||!meta.has_detections||!meta.pieces){vd.removeAttribute('src');
  $('#msg').textContent='The video or the detection file of this segment is not on this machine. Skip it, or restart with --refill.';return}
 vd.src='/video/'+encodeURIComponent(meta.row.video_id);vd.onloadedmetadata=()=>{startPiece(0);vd.play()}}
async function save(status){const it=items[idx];const rec={status,one_object:picked(),notes:$('#nt').value};
 if(status==='done'&&!rec.one_object){alert('Please answer: is it the same object all the time?');return}
 await jpost('/api/tracks/'+encodeURIComponent(it.id),rec);it.status=status;progress();go('tracks',show,1)}
document.addEventListener('keydown',e=>{if(/TEXTAREA|INPUT/.test(e.target.tagName)&&e.target.type!=='radio')return;
 if(e.key===' '){e.preventDefault();toggle()}else if(e.key==='Enter')save('done');else if(e.key==='z')zoomed=!zoomed;
 else if(e.key==='r')restart();else if('1234'.includes(e.key)&&meta)setPicked(KEYS[+e.key-1])});
requestAnimationFrame(draw);start('tracks',show);
