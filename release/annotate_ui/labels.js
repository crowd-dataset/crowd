let row={},player=null,ready=false,pending=null,local=false,lv=null,hasLocal=false;
function onYouTubeIframeAPIReady(){ready=true;if(pending){const r=pending;pending=null;loadYT(r)}}
function loadYT(r){if(!ready){pending=r;return}
 if(!player){player=new YT.Player('player',{width:'100%',height:'420',videoId:r.video_id,
  playerVars:{start:Math.floor(r.start_s),rel:0},events:{onReady:()=>seek(row.start_s),onError:ytError}})}
 else player.loadVideoById({videoId:r.video_id,startSeconds:r.start_s})}
function ytError(){$('#src').innerHTML='<b style="color:#b00020">This video is not available on YouTube</b> and there is no '+
 'downloaded copy on this machine: skip it.'}
let stopAt=null;
function seek(t){stopAt=null;if(local){if(lv)lv.currentTime=t}else if(player&&player.seekTo)player.seekTo(t,true)}
function now(){return local?(lv?lv.currentTime:0):(player&&player.getCurrentTime?player.getCurrentTime():0)}
function pauseNow(){if(local){if(lv)lv.pause()}else if(player&&player.pauseVideo)player.pauseVideo()}
function playNow(){if(local){if(lv)lv.play()}else if(player&&player.playVideo)player.playVideo()}
function playSegment(){seek(row.start_s);stopAt=row.end_s;playNow()}
function fmt(t){t=Math.max(0,Math.floor(t));const h=Math.floor(t/3600),m=Math.floor(t%3600/60),s=t%60;  // as the player shows it
 return (h?h+':'+String(m).padStart(2,'0'):m)+':'+String(s).padStart(2,'0')}
function shown(v){return v===undefined||v===''?'':(Number.isInteger(+v)?fmt(+v):String(v))}
function both(t){return fmt(t)+' ('+t+' s)'}
function toSec(x){x=String(x).trim();if(x==='')return '';if(/^\d+(\.\d+)?$/.test(x))return x;  // seconds as typed
 const p=x.split(':');if(p.length>3||p.some(v=>!/^\d+$/.test(v)))return null;return String(p.reduce((a,v)=>a*60+parseInt(v,10),0))}
setInterval(()=>{if(typeof row.start_s!=='number')return;const t=now(),len=Math.round(row.end_s-row.start_s);
 const inside=t>=row.start_s-0.5&&t<=row.end_s+0.5;
 $('#clock').innerHTML=(inside?'':'<b style="color:#b00020">Outside the segment.</b> ')+'Now '+fmt(t)+' &middot; the segment is '+
  fmt(row.start_s)+' to '+fmt(row.end_s)+' ('+len+' s)'+(inside?' &middot; second '+Math.max(0,Math.round(t-row.start_s))+' of '+len:'');
},250);
let prev=0;  // the video always stops where the segment ends; going past it by seeking is still possible
setInterval(()=>{const t=now(),end=row.end_s;
 if(typeof end==='number'&&((stopAt!==null&&t>=stopAt)||(prev<end&&t>=end&&t-prev<1))){pauseNow();stopAt=null;seek(end)}
 prev=t},50);
function alt(){const b=$('#alt');b.style.display=hasLocal?'inline-block':'none';
 b.textContent=local?'Use YouTube instead':'Use the downloaded file';b.onclick=()=>local?useYouTube(row):useLocal(row)}
function useLocal(r){local=true;$('#player').style.display='none';if(player&&player.pauseVideo)player.pauseVideo();
 const box=$('#lvbox');box.style.display='block';
 box.innerHTML='<video id="lv" controls preload="metadata" style="width:100%;max-height:520px;background:#000"></video>';
 lv=$('#lv');lv.onloadedmetadata=()=>{lv.currentTime=r.start_s};lv.src='/video/'+encodeURIComponent(r.video_id);
 $('#src').textContent='Playing the downloaded file from this machine (the one the detector processed).';alt()}
function useYouTube(r){local=false;if(lv)lv.pause();$('#lvbox').style.display='none';$('#lvbox').innerHTML='';lv=null;
 $('#player').style.display='block';
 $('#src').textContent=hasLocal?'Playing from YouTube.':'No downloaded copy on this machine: playing from YouTube.';alt();loadYT(r)}
document.querySelectorAll('input[name=b]').forEach(e=>e.onchange=()=>{$('#fix').style.display=e.value==='n'&&e.checked?'block':'none'});
async function show(){const it=items[idx];const d=await jget('/api/labels/'+encodeURIComponent(it.id));row=d.row;
 row.start_s=parseFloat(row.start_s);row.end_s=parseFloat(row.end_s);
 $('#pf').innerHTML='<option value="">-</option>'+d.platforms.map(p=>'<option>'+p+'</option>').join('')+'<option>unclear</option>';
 $('#pos').textContent=(idx+1)+' / '+items.length+' ('+it.status+')';$('#seg').textContent=row.segment_id;
 $('#st').textContent=both(row.start_s);$('#en').textContent=both(row.end_s);$('#yt').href=row.url_start;
 $('#by').textContent=row.contributor;$('#same').textContent=row.contributor.toLowerCase()===(d.annotator||'').toLowerCase()?
  ' (you: an independent person should audit this segment, please skip it)':'';
 const s=d.saved||{};document.querySelectorAll('input[name=b]').forEach(e=>e.checked=s.boundary_ok===e.value);
 $('#fix').style.display=s.boundary_ok==='n'?'block':'none';$('#cs').value=shown(s.correct_start_s);$('#ce').value=shown(s.correct_end_s);
 $('#tod').value=s.tod_audit||'';$('#pf').value=s.platform_audit||'';$('#nt').value=s.notes||'';hasLocal=!!d.local;if(hasLocal)useLocal(row);else useYouTube(row)}
async function save(status){const it=items[idx];const b=document.querySelector('input[name=b]:checked');
 if(status==='done'&&(!b||!$('#tod').value||!$('#pf').value)){alert('Please answer the boundaries, the time of day and the platform.');return}
 const cs=toSec($('#cs').value),ce=toSec($('#ce').value);
 if(status==='done'&&(cs===null||ce===null)){alert('Please type the corrected times like 6:55 or 1:02:05, or in seconds.');return}
 const rec={status,boundary_ok:b?b.value:'',correct_start_s:cs===null?'':cs,correct_end_s:ce===null?'':ce,
  tod_audit:$('#tod').value,platform_audit:$('#pf').value,notes:$('#nt').value};
 await jpost('/api/labels/'+encodeURIComponent(it.id),rec);it.status=status;progress();go('labels',show,1)}
document.addEventListener('keydown',e=>{if(e.key==='Enter'&&!/TEXTAREA/.test(e.target.tagName))save('done')});
const tag=document.createElement('script');tag.src='https://www.youtube.com/iframe_api';document.head.appendChild(tag);
start('labels',show);
