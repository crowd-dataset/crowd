"""Check caption_hit on texts EasyOCR read in the openings of the mapping's videos (2026-10-22): captions of
highlight montages, misread ones too, and the signs and sentences the old frame-wide search took for them.
Run from the repo with the crowd env's python."""
import propose_segments as ps

captions = [['HIGHLIGHT SCENES'], ['Video highlights_'], ['IN THIS VIDEO:'], ['In this', 'video..'], ['COMING UP'],
            ['Coming Up Next:'], ['A SHORT PREVIEW'], ['Scenes from the Drive'], ['HIGHIIGHT SCENES'],
            ['Seoul 4K Highlights'], ['coming upnext'], ['inthis'], ['TRAILER'], ['KFC', 'Preview']]
not_captions = [['Highland Av'], ['N Highland Ave'], ['highlanders'], ['trailers'],
                ['The Texas A&M Transportation Institute conducted the study, which highlights how traffic'],
                ['talking and playing radio in this video.'], ['subscribe if you like this video'],
                ['in this video we are driving to the city centre'], ['Koh Pich; Phnom Penh City'], []]
for texts in captions:
    assert ps.caption_hit(texts), texts
for texts in not_captions:
    assert not ps.caption_hit(texts), texts
print('ok')
