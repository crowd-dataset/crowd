let row={},player,ready=false,pending=null;
function onYouTubeIframeAPIReady(){ready=true;if(pending)load(pending)}
function load(r){if(!ready){pending=r;return}
 if(!player){player=new YT.Player('player',{width:'100%',height:'420',videoId:r.video_id,
  playerVars:{start:Math.floor(r.start_s),rel:0},events:{onReady:()=>seek(row.start_s)}})}
 else player.loadVideoById({videoId:r.video_id,startSeconds:r.start_s})}
function seek(t){if(player)player.seekTo(t,true)}
document.querySelectorAll('input[name=b]').forEach(e=>e.onchange=()=>{$('#fix').style.display=e.value==='n'&&e.checked?'block':'none'});
async function show(){const it=items[idx];const d=await jget('/api/labels/'+encodeURIComponent(it.id));row=d.row;
 row.start_s=parseFloat(row.start_s);row.end_s=parseFloat(row.end_s);
 $('#pf').innerHTML='<option value="">-</option>'+d.platforms.map(p=>'<option>'+p+'</option>').join('')+'<option>unclear</option>';
 $('#pos').textContent=(idx+1)+' / '+items.length+' ('+it.status+')';$('#seg').textContent=row.segment_id;
 $('#st').textContent=row.start_s;$('#en').textContent=row.end_s;$('#yt').href=row.url_start;
 $('#by').textContent=row.contributor;$('#same').textContent=row.contributor.toLowerCase()===(d.annotator||'').toLowerCase()?
  ' (you: an independent person should audit this segment, please skip it)':'';
 const s=d.saved||{};document.querySelectorAll('input[name=b]').forEach(e=>e.checked=s.boundary_ok===e.value);
 $('#fix').style.display=s.boundary_ok==='n'?'block':'none';$('#cs').value=s.correct_start_s||'';$('#ce').value=s.correct_end_s||'';
 $('#tod').value=s.tod_audit||'';$('#pf').value=s.platform_audit||'';$('#nt').value=s.notes||'';load(row)}
async function save(status){const it=items[idx];const b=document.querySelector('input[name=b]:checked');
 if(status==='done'&&(!b||!$('#tod').value||!$('#pf').value)){alert('Please answer the boundaries, the time of day and the platform.');return}
 const rec={status,boundary_ok:b?b.value:'',correct_start_s:$('#cs').value,correct_end_s:$('#ce').value,
  tod_audit:$('#tod').value,platform_audit:$('#pf').value,notes:$('#nt').value};
 await jpost('/api/labels/'+encodeURIComponent(it.id),rec);it.status=status;progress();go('labels',show,1)}
document.addEventListener('keydown',e=>{if(e.key==='Enter'&&!/TEXTAREA/.test(e.target.tagName))save('done')});
const tag=document.createElement('script');tag.src='https://www.youtube.com/iframe_api';document.head.appendChild(tag);
start('labels',show);
