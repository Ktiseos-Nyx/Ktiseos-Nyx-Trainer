(()=>{var a={};a.id=40,a.ids=[40],a.modules={261:a=>{"use strict";a.exports=require("next/dist/shared/lib/router/utils/app-paths")},3295:a=>{"use strict";a.exports=require("next/dist/server/app-render/after-task-async-storage.external.js")},10846:a=>{"use strict";a.exports=require("next/dist/compiled/next-server/app-page.runtime.prod.js")},19121:a=>{"use strict";a.exports=require("next/dist/server/app-render/action-async-storage.external.js")},26713:a=>{"use strict";a.exports=require("next/dist/shared/lib/router/utils/is-bot")},28217:(a,b,c)=>{Promise.resolve().then(c.bind(c,32524))},28354:a=>{"use strict";a.exports=require("util")},29294:a=>{"use strict";a.exports=require("next/dist/server/app-render/work-async-storage.external.js")},32524:(a,b,c)=>{"use strict";c.r(b),c.d(b,{default:()=>m});var d=c(21124),e=c(38301),f=c(58562),g=c(74097),h=c(78583),i=c(59405),j=c(90166),k=c(3378);let l=[{id:"getting-started",title:"Getting Started",icon:"\uD83D\uDE80",content:`Welcome to Ktiseos-Nyx Trainer! This guide will help you get started with training your first LoRA model.

**Prerequisites:**
- A dataset of images (20-100+ images recommended)
- A base model downloaded (SDXL, SD 1.5, Flux, or SD 3.5)
- (Optional) A VAE file for improved quality

**Quick Start Workflow:**
1. Download a base model from the Models & VAEs page
2. Upload your dataset (images in a folder)
3. Auto-tag your images using WD14 tagger
4. Configure training parameters
5. Start training and monitor progress
6. Download your trained LoRA`,links:[{label:"VastAI Setup Guide",url:"https://vast.ai/docs"},{label:"Kohya Documentation",url:"https://github.com/kohya-ss/sd-scripts"}]},{id:"datasets",title:"Dataset Preparation",icon:"\uD83D\uDCC1",content:`Proper dataset preparation is crucial for good LoRA training results.

**Dataset Structure:**
- Use Kohya format: \`{repeats}_{folder_name}\`
- Example: \`10_my_character\` (10 repeats of images in folder)
- Higher repeats = more training on those images

**Image Requirements:**
- Format: JPG, PNG, or WebP
- Resolution: 512x512 to 1024x1024 (SDXL prefers 1024x1024)
- Quality: High-quality, well-lit images work best
- Variety: Different poses, angles, and settings
- Quantity: 20-100 images is a good range

**Tagging:**
- WD14 tagger auto-generates tags for each image
- Tags are saved as .txt files next to images
- You can manually edit tags in the Tag Editor
- Common format: \`1girl, blue eyes, long hair, smile\``,links:[{label:"WD14 Tagger Models",url:"https://huggingface.co/SmilingWolf"}]},{id:"training-config",title:"Training Configuration",icon:"⚙️",content:`Understanding training parameters helps you achieve better results.

**Essential Parameters:**

**Learning Rate (LR):**
- Default: 1e-4 (0.0001)
- Lower = slower, more stable training
- Higher = faster, but risk of overtraining
- SDXL: 1e-4 to 5e-5
- SD 1.5: 1e-4 to 1e-3

**Network Rank (Dim):**
- Default: 32
- Higher = more detail, larger file size
- Common values: 8, 16, 32, 64, 128
- Recommendation: 32 for characters, 64+ for styles

**Epochs:**
- Number of times to train on full dataset
- Start with 5-10 epochs
- Monitor loss to avoid overtraining

**Batch Size:**
- Higher = faster training, more VRAM
- Lower = slower, less VRAM
- Recommendation: 1-4 depending on GPU`},{id:"lora-types",title:"LoRA Types (LyCORIS)",icon:"\uD83D\uDD27",content:`Different LoRA architectures for different use cases.

**Standard LoRA:**
- Most common and widely supported
- Good for characters and concepts
- Efficient and fast to train

**LoCon (LoRA for Convolution):**
- Better for style transfer
- Captures texture and patterns well
- Slightly larger file size

**LoHa (LoRA with Hadamard Product):**
- Advanced architecture
- Better detail retention
- More complex training

**LoKr (LoRA with Kronecker Product):**
- Experimental but powerful
- Can capture fine details
- Requires more experimentation

**Recommendation:** Start with standard LoRA, experiment with LoCon for styles.`},{id:"monitoring",title:"Training Monitoring",icon:"\uD83D\uDCCA",content:`Understanding training progress and when to stop.

**Loss Values:**
- Loss should generally decrease over time
- Too low (<0.05): Possible overtraining
- Fluctuating wildly: Learning rate too high
- Not decreasing: Learning rate too low

**Sample Images:**
- Generate sample images during training
- Check for quality and adherence to concept
- Stop if images degrade (overtraining)

**Signs of Good Training:**
- Steady loss decrease
- Sample images improve quality
- Concept is recognizable
- No artifacts or degradation

**Signs of Overtraining:**
- Loss becomes very low (<0.05)
- Images look "burned" or have artifacts
- Loss of diversity in outputs
- Stop training and use earlier checkpoint`},{id:"post-training",title:"Post-Training",icon:"\uD83C\uDFA8",content:`What to do after training completes.

**LoRA Resizing:**
- Reduce LoRA file size while maintaining quality
- Useful for sharing or reducing VRAM usage
- Common targets: dim 32 → 16 or 8
- Use the Resize tool in Utilities

**Testing Your LoRA:**
- Test in multiple inference UIs (ComfyUI, A1111, Forge)
- Try different strengths (0.5 to 1.0)
- Combine with different base models
- Test with various prompts

**Sharing:**
- Upload to HuggingFace Hub via Utilities page
- Include sample images and trigger words
- Document recommended settings
- Specify compatible base models`},{id:"troubleshooting",title:"Troubleshooting",icon:"\uD83D\uDD0D",content:`Common issues and solutions.

**Training Won't Start:**
- Check that base model is downloaded
- Verify dataset path is correct
- Ensure images are valid format
- Check backend logs for errors

**Out of Memory (OOM):**
- Reduce batch size to 1
- Lower network dimension (rank)
- Enable gradient checkpointing
- Use a smaller base model

**Poor Results:**
- Increase number of epochs
- Adjust learning rate
- Add more varied training images
- Check image quality and tags
- Try different LoRA type

**Loss Not Decreasing:**
- Increase learning rate
- Check that images are properly tagged
- Verify dataset structure
- Increase batch size if possible`}];function m(){let[a,b]=(0,e.useState)("getting-started"),c=l.find(b=>b.id===a)||l[0];return(0,d.jsx)("div",{className:"min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900 py-16",children:(0,d.jsxs)("div",{className:"container mx-auto px-4 max-w-7xl",children:[(0,d.jsx)(j.A,{items:[{label:"Home",href:"/",icon:(0,d.jsx)(f.A,{className:"w-4 h-4"})},{label:"Documentation",icon:(0,d.jsx)(g.A,{className:"w-4 h-4"})}]}),(0,d.jsxs)("div",{className:"mb-8",children:[(0,d.jsx)("h1",{className:"text-5xl md:text-6xl font-bold mb-4 bg-gradient-to-r from-green-400 via-emerald-400 to-green-400 bg-clip-text text-transparent",children:"Documentation"}),(0,d.jsx)("p",{className:"text-xl text-gray-300",children:"Guides and references for LoRA training"})]}),(0,d.jsxs)("div",{className:"grid lg:grid-cols-4 gap-6",children:[(0,d.jsx)("div",{className:"lg:col-span-1",children:(0,d.jsxs)("div",{className:"bg-slate-800/50 backdrop-blur-sm border border-slate-700 rounded-lg p-4 sticky top-4",children:[(0,d.jsx)("h2",{className:"text-lg font-bold text-white mb-4",children:"Sections"}),(0,d.jsx)("nav",{className:"space-y-1",children:l.map(c=>(0,d.jsxs)("button",{onClick:()=>b(c.id),className:`w-full flex items-center gap-3 px-3 py-2 rounded-lg text-left transition-colors ${a===c.id?"bg-cyan-500/20 text-cyan-400 border border-cyan-500/30":"text-gray-400 hover:text-gray-300 hover:bg-slate-700/50"}`,children:[(0,d.jsx)("span",{className:"text-xl",children:c.icon}),(0,d.jsx)("span",{className:"text-sm font-medium",children:c.title})]},c.id))})]})}),(0,d.jsx)("div",{className:"lg:col-span-3",children:(0,d.jsx)(k.hX,{variant:"ocean",intensity:"subtle",children:(0,d.jsxs)("div",{className:"p-8",children:[(0,d.jsxs)("div",{className:"flex items-center gap-3 mb-6",children:[(0,d.jsx)("span",{className:"text-4xl",children:c.icon}),(0,d.jsx)("h2",{className:"text-3xl font-bold text-white",children:c.title})]}),(0,d.jsx)("div",{className:"prose prose-invert max-w-none",children:c.content.split("\n\n").map((a,b)=>{if(a.startsWith("**")&&a.endsWith(":**"))return(0,d.jsx)("h3",{className:"text-xl font-bold text-cyan-400 mt-6 mb-3",children:a.replace(/\*\*/g,"").replace(":","")},b);if(a.startsWith("**")&&a.includes("**\n")){let[c,...e]=a.split("\n");return(0,d.jsxs)("div",{className:"mb-4",children:[(0,d.jsx)("h4",{className:"text-lg font-semibold text-white mb-2",children:c.replace(/\*\*/g,"")}),(0,d.jsx)("p",{className:"text-gray-300 leading-relaxed whitespace-pre-line",children:e.join("\n")})]},b)}if(!(a.startsWith("- ")||a.includes("\n- ")))return(0,d.jsx)("p",{className:"text-gray-300 leading-relaxed mb-4",dangerouslySetInnerHTML:{__html:a.replace(/\*\*(.*?)\*\*/g,'<strong class="text-white">$1</strong>').replace(/`(.*?)`/g,'<code class="bg-slate-800 px-2 py-1 rounded text-cyan-400">$1</code>')}},b);{let c=a.split("\n").filter(a=>a.trim());return(0,d.jsx)("ul",{className:"list-disc list-inside space-y-2 text-gray-300 mb-4",children:c.map((a,b)=>(0,d.jsx)("li",{className:"ml-4",children:a.replace(/^- /,"").replace(/\*\*(.*?)\*\*/g,'<strong class="text-white">$1</strong>')},b))},b)}})}),c.links&&c.links.length>0&&(0,d.jsxs)("div",{className:"mt-8 pt-6 border-t border-slate-700",children:[(0,d.jsx)("h4",{className:"text-lg font-semibold text-white mb-3",children:"External Resources"}),(0,d.jsx)("div",{className:"space-y-2",children:c.links.map((a,b)=>(0,d.jsxs)("a",{href:a.url,target:"_blank",rel:"noopener noreferrer",className:"flex items-center gap-2 text-cyan-400 hover:text-cyan-300 transition-colors",children:[(0,d.jsx)(h.A,{className:"w-4 h-4"}),(0,d.jsx)("span",{children:a.label}),(0,d.jsx)(i.A,{className:"w-4 h-4"})]},b))})]})]})})})]})]})})}},33873:a=>{"use strict";a.exports=require("path")},41025:a=>{"use strict";a.exports=require("next/dist/server/app-render/dynamic-access-async-storage.external.js")},42416:(a,b,c)=>{"use strict";c.r(b),c.d(b,{GlobalError:()=>D.a,__next_app__:()=>J,handler:()=>L,pages:()=>I,routeModule:()=>K,tree:()=>H});var d=c(49754),e=c(9117),f=c(46595),g=c(32324),h=c(39326),i=c(38928),j=c(20175),k=c(12),l=c(54290),m=c(12696),n=c(82802),o=c(77533),p=c(45229),q=c(32822),r=c(261),s=c(26453),t=c(52474),u=c(26713),v=c(51356),w=c(62685),x=c(36225),y=c(63446),z=c(2762),A=c(45742),B=c(86439),C=c(81170),D=c.n(C),E=c(62506),F=c(91203),G={};for(let a in E)0>["default","tree","pages","GlobalError","__next_app__","routeModule","handler"].indexOf(a)&&(G[a]=()=>E[a]);c.d(b,G);let H={children:["",{children:["docs",{children:["__PAGE__",{},{page:[()=>Promise.resolve().then(c.bind(c,59798)),"/Users/duskfall/Ktiseos-Nyx-Trainer/frontend/app/docs/page.tsx"]}]},{}]},{layout:[()=>Promise.resolve().then(c.bind(c,16953)),"/Users/duskfall/Ktiseos-Nyx-Trainer/frontend/app/layout.tsx"],"global-error":[()=>Promise.resolve().then(c.t.bind(c,81170,23)),"next/dist/client/components/builtin/global-error.js"],"not-found":[()=>Promise.resolve().then(c.t.bind(c,87028,23)),"next/dist/client/components/builtin/not-found.js"],forbidden:[()=>Promise.resolve().then(c.t.bind(c,90461,23)),"next/dist/client/components/builtin/forbidden.js"],unauthorized:[()=>Promise.resolve().then(c.t.bind(c,32768,23)),"next/dist/client/components/builtin/unauthorized.js"]}]}.children,I=["/Users/duskfall/Ktiseos-Nyx-Trainer/frontend/app/docs/page.tsx"],J={require:c,loadChunk:()=>Promise.resolve()},K=new d.AppPageRouteModule({definition:{kind:e.RouteKind.APP_PAGE,page:"/docs/page",pathname:"/docs",bundlePath:"",filename:"",appPaths:[]},userland:{loaderTree:H},distDir:".next",relativeProjectDir:""});async function L(a,b,d){var C;let G="/docs/page";"/index"===G&&(G="/");let M=(0,h.getRequestMeta)(a,"postponed"),N=(0,h.getRequestMeta)(a,"minimalMode"),O=await K.prepare(a,b,{srcPage:G,multiZoneDraftMode:!1});if(!O)return b.statusCode=400,b.end("Bad Request"),null==d.waitUntil||d.waitUntil.call(d,Promise.resolve()),null;let{buildId:P,query:Q,params:R,parsedUrl:S,pageIsDynamic:T,buildManifest:U,nextFontManifest:V,reactLoadableManifest:W,serverActionsManifest:X,clientReferenceManifest:Y,subresourceIntegrityManifest:Z,prerenderManifest:$,isDraftMode:_,resolvedPathname:aa,revalidateOnlyGenerated:ab,routerServerContext:ac,nextConfig:ad,interceptionRoutePatterns:ae}=O,af=S.pathname||"/",ag=(0,r.normalizeAppPath)(G),{isOnDemandRevalidate:ah}=O,ai=K.match(af,$),aj=!!$.routes[aa],ak=!!(ai||aj||$.routes[ag]),al=a.headers["user-agent"]||"",am=(0,u.getBotType)(al),an=(0,p.isHtmlBotRequest)(a),ao=(0,h.getRequestMeta)(a,"isPrefetchRSCRequest")??"1"===a.headers[t.NEXT_ROUTER_PREFETCH_HEADER],ap=(0,h.getRequestMeta)(a,"isRSCRequest")??!!a.headers[t.RSC_HEADER],aq=(0,s.getIsPossibleServerAction)(a),ar=(0,m.checkIsAppPPREnabled)(ad.experimental.ppr)&&(null==(C=$.routes[ag]??$.dynamicRoutes[ag])?void 0:C.renderingMode)==="PARTIALLY_STATIC",as=!1,at=!1,au=ar?M:void 0,av=ar&&ap&&!ao,aw=(0,h.getRequestMeta)(a,"segmentPrefetchRSCRequest"),ax=!al||(0,p.shouldServeStreamingMetadata)(al,ad.htmlLimitedBots);an&&ar&&(ak=!1,ax=!1);let ay=!0===K.isDev||!ak||"string"==typeof M||av,az=an&&ar,aA=null;_||!ak||ay||aq||au||av||(aA=aa);let aB=aA;!aB&&K.isDev&&(aB=aa),K.isDev||_||!ak||!ap||av||(0,k.d)(a.headers);let aC={...E,tree:H,pages:I,GlobalError:D(),handler:L,routeModule:K,__next_app__:J};X&&Y&&(0,o.setReferenceManifestsSingleton)({page:G,clientReferenceManifest:Y,serverActionsManifest:X,serverModuleMap:(0,q.createServerModuleMap)({serverActionsManifest:X})});let aD=a.method||"GET",aE=(0,g.getTracer)(),aF=aE.getActiveScopeSpan();try{let f=K.getVaryHeader(aa,ae);b.setHeader("Vary",f);let k=async(c,d)=>{let e=new l.NodeNextRequest(a),f=new l.NodeNextResponse(b);return K.render(e,f,d).finally(()=>{if(!c)return;c.setAttributes({"http.status_code":b.statusCode,"next.rsc":!1});let d=aE.getRootSpanAttributes();if(!d)return;if(d.get("next.span_type")!==i.BaseServerSpan.handleRequest)return void console.warn(`Unexpected root span type '${d.get("next.span_type")}'. Please report this Next.js issue https://github.com/vercel/next.js`);let e=d.get("next.route");if(e){let a=`${aD} ${e}`;c.setAttributes({"next.route":e,"http.route":e,"next.span_name":a}),c.updateName(a)}else c.updateName(`${aD} ${a.url}`)})},m=async({span:e,postponed:f,fallbackRouteParams:g})=>{let i={query:Q,params:R,page:ag,sharedContext:{buildId:P},serverComponentsHmrCache:(0,h.getRequestMeta)(a,"serverComponentsHmrCache"),fallbackRouteParams:g,renderOpts:{App:()=>null,Document:()=>null,pageConfig:{},ComponentMod:aC,Component:(0,j.T)(aC),params:R,routeModule:K,page:G,postponed:f,shouldWaitOnAllReady:az,serveStreamingMetadata:ax,supportsDynamicResponse:"string"==typeof f||ay,buildManifest:U,nextFontManifest:V,reactLoadableManifest:W,subresourceIntegrityManifest:Z,serverActionsManifest:X,clientReferenceManifest:Y,setIsrStatus:null==ac?void 0:ac.setIsrStatus,dir:c(33873).join(process.cwd(),K.relativeProjectDir),isDraftMode:_,isRevalidate:ak&&!f&&!av,botType:am,isOnDemandRevalidate:ah,isPossibleServerAction:aq,assetPrefix:ad.assetPrefix,nextConfigOutput:ad.output,crossOrigin:ad.crossOrigin,trailingSlash:ad.trailingSlash,previewProps:$.preview,deploymentId:ad.deploymentId,enableTainting:ad.experimental.taint,htmlLimitedBots:ad.htmlLimitedBots,devtoolSegmentExplorer:ad.experimental.devtoolSegmentExplorer,reactMaxHeadersLength:ad.reactMaxHeadersLength,multiZoneDraftMode:!1,incrementalCache:(0,h.getRequestMeta)(a,"incrementalCache"),cacheLifeProfiles:ad.experimental.cacheLife,basePath:ad.basePath,serverActions:ad.experimental.serverActions,...as?{nextExport:!0,supportsDynamicResponse:!1,isStaticGeneration:!0,isRevalidate:!0,isDebugDynamicAccesses:as}:{},experimental:{isRoutePPREnabled:ar,expireTime:ad.expireTime,staleTimes:ad.experimental.staleTimes,cacheComponents:!!ad.experimental.cacheComponents,clientSegmentCache:!!ad.experimental.clientSegmentCache,clientParamParsing:!!ad.experimental.clientParamParsing,dynamicOnHover:!!ad.experimental.dynamicOnHover,inlineCss:!!ad.experimental.inlineCss,authInterrupts:!!ad.experimental.authInterrupts,clientTraceMetadata:ad.experimental.clientTraceMetadata||[]},waitUntil:d.waitUntil,onClose:a=>{b.on("close",a)},onAfterTaskError:()=>{},onInstrumentationRequestError:(b,c,d)=>K.onRequestError(a,b,d,ac),err:(0,h.getRequestMeta)(a,"invokeError"),dev:K.isDev}},l=await k(e,i),{metadata:m}=l,{cacheControl:n,headers:o={},fetchTags:p}=m;if(p&&(o[y.NEXT_CACHE_TAGS_HEADER]=p),a.fetchMetrics=m.fetchMetrics,ak&&(null==n?void 0:n.revalidate)===0&&!K.isDev&&!ar){let a=m.staticBailoutInfo,b=Object.defineProperty(Error(`Page changed from static to dynamic at runtime ${aa}${(null==a?void 0:a.description)?`, reason: ${a.description}`:""}
see more here https://nextjs.org/docs/messages/app-static-to-dynamic-error`),"__NEXT_ERROR_CODE",{value:"E132",enumerable:!1,configurable:!0});if(null==a?void 0:a.stack){let c=a.stack;b.stack=b.message+c.substring(c.indexOf("\n"))}throw b}return{value:{kind:v.CachedRouteKind.APP_PAGE,html:l,headers:o,rscData:m.flightData,postponed:m.postponed,status:m.statusCode,segmentData:m.segmentData},cacheControl:n}},o=async({hasResolved:c,previousCacheEntry:f,isRevalidating:g,span:i})=>{let j,k=!1===K.isDev,l=c||b.writableEnded;if(ah&&ab&&!f&&!N)return(null==ac?void 0:ac.render404)?await ac.render404(a,b):(b.statusCode=404,b.end("This page could not be found")),null;if(ai&&(j=(0,w.parseFallbackField)(ai.fallback)),j===w.FallbackMode.PRERENDER&&(0,u.isBot)(al)&&(!ar||an)&&(j=w.FallbackMode.BLOCKING_STATIC_RENDER),(null==f?void 0:f.isStale)===-1&&(ah=!0),ah&&(j!==w.FallbackMode.NOT_FOUND||f)&&(j=w.FallbackMode.BLOCKING_STATIC_RENDER),!N&&j!==w.FallbackMode.BLOCKING_STATIC_RENDER&&aB&&!l&&!_&&T&&(k||!aj)){let b;if((k||ai)&&j===w.FallbackMode.NOT_FOUND)throw new B.NoFallbackError;if(ar&&!ap){let c="string"==typeof(null==ai?void 0:ai.fallback)?ai.fallback:k?ag:null;if(b=await K.handleResponse({cacheKey:c,req:a,nextConfig:ad,routeKind:e.RouteKind.APP_PAGE,isFallback:!0,prerenderManifest:$,isRoutePPREnabled:ar,responseGenerator:async()=>m({span:i,postponed:void 0,fallbackRouteParams:k||at?(0,n.u)(ag):null}),waitUntil:d.waitUntil}),null===b)return null;if(b)return delete b.cacheControl,b}}let o=ah||g||!au?void 0:au;if(as&&void 0!==o)return{cacheControl:{revalidate:1,expire:void 0},value:{kind:v.CachedRouteKind.PAGES,html:x.default.EMPTY,pageData:{},headers:void 0,status:void 0}};let p=T&&ar&&((0,h.getRequestMeta)(a,"renderFallbackShell")||at)?(0,n.u)(af):null;return m({span:i,postponed:o,fallbackRouteParams:p})},p=async c=>{var f,g,i,j,k;let l,n=await K.handleResponse({cacheKey:aA,responseGenerator:a=>o({span:c,...a}),routeKind:e.RouteKind.APP_PAGE,isOnDemandRevalidate:ah,isRoutePPREnabled:ar,req:a,nextConfig:ad,prerenderManifest:$,waitUntil:d.waitUntil});if(_&&b.setHeader("Cache-Control","private, no-cache, no-store, max-age=0, must-revalidate"),K.isDev&&b.setHeader("Cache-Control","no-store, must-revalidate"),!n){if(aA)throw Object.defineProperty(Error("invariant: cache entry required but not generated"),"__NEXT_ERROR_CODE",{value:"E62",enumerable:!1,configurable:!0});return null}if((null==(f=n.value)?void 0:f.kind)!==v.CachedRouteKind.APP_PAGE)throw Object.defineProperty(Error(`Invariant app-page handler received invalid cache entry ${null==(i=n.value)?void 0:i.kind}`),"__NEXT_ERROR_CODE",{value:"E707",enumerable:!1,configurable:!0});let p="string"==typeof n.value.postponed;ak&&!av&&(!p||ao)&&(N||b.setHeader("x-nextjs-cache",ah?"REVALIDATED":n.isMiss?"MISS":n.isStale?"STALE":"HIT"),b.setHeader(t.NEXT_IS_PRERENDER_HEADER,"1"));let{value:q}=n;if(au)l={revalidate:0,expire:void 0};else if(N&&ap&&!ao&&ar)l={revalidate:0,expire:void 0};else if(!K.isDev)if(_)l={revalidate:0,expire:void 0};else if(ak){if(n.cacheControl)if("number"==typeof n.cacheControl.revalidate){if(n.cacheControl.revalidate<1)throw Object.defineProperty(Error(`Invalid revalidate configuration provided: ${n.cacheControl.revalidate} < 1`),"__NEXT_ERROR_CODE",{value:"E22",enumerable:!1,configurable:!0});l={revalidate:n.cacheControl.revalidate,expire:(null==(j=n.cacheControl)?void 0:j.expire)??ad.expireTime}}else l={revalidate:y.CACHE_ONE_YEAR,expire:void 0}}else b.getHeader("Cache-Control")||(l={revalidate:0,expire:void 0});if(n.cacheControl=l,"string"==typeof aw&&(null==q?void 0:q.kind)===v.CachedRouteKind.APP_PAGE&&q.segmentData){b.setHeader(t.NEXT_DID_POSTPONE_HEADER,"2");let c=null==(k=q.headers)?void 0:k[y.NEXT_CACHE_TAGS_HEADER];N&&ak&&c&&"string"==typeof c&&b.setHeader(y.NEXT_CACHE_TAGS_HEADER,c);let d=q.segmentData.get(aw);return void 0!==d?(0,A.sendRenderResult)({req:a,res:b,generateEtags:ad.generateEtags,poweredByHeader:ad.poweredByHeader,result:x.default.fromStatic(d,t.RSC_CONTENT_TYPE_HEADER),cacheControl:n.cacheControl}):(b.statusCode=204,(0,A.sendRenderResult)({req:a,res:b,generateEtags:ad.generateEtags,poweredByHeader:ad.poweredByHeader,result:x.default.EMPTY,cacheControl:n.cacheControl}))}let r=(0,h.getRequestMeta)(a,"onCacheEntry");if(r&&await r({...n,value:{...n.value,kind:"PAGE"}},{url:(0,h.getRequestMeta)(a,"initURL")}))return null;if(p&&au)throw Object.defineProperty(Error("Invariant: postponed state should not be present on a resume request"),"__NEXT_ERROR_CODE",{value:"E396",enumerable:!1,configurable:!0});if(q.headers){let a={...q.headers};for(let[c,d]of(N&&ak||delete a[y.NEXT_CACHE_TAGS_HEADER],Object.entries(a)))if(void 0!==d)if(Array.isArray(d))for(let a of d)b.appendHeader(c,a);else"number"==typeof d&&(d=d.toString()),b.appendHeader(c,d)}let s=null==(g=q.headers)?void 0:g[y.NEXT_CACHE_TAGS_HEADER];if(N&&ak&&s&&"string"==typeof s&&b.setHeader(y.NEXT_CACHE_TAGS_HEADER,s),!q.status||ap&&ar||(b.statusCode=q.status),!N&&q.status&&F.RedirectStatusCode[q.status]&&ap&&(b.statusCode=200),p&&b.setHeader(t.NEXT_DID_POSTPONE_HEADER,"1"),ap&&!_){if(void 0===q.rscData){if(q.postponed)throw Object.defineProperty(Error("Invariant: Expected postponed to be undefined"),"__NEXT_ERROR_CODE",{value:"E372",enumerable:!1,configurable:!0});return(0,A.sendRenderResult)({req:a,res:b,generateEtags:ad.generateEtags,poweredByHeader:ad.poweredByHeader,result:q.html,cacheControl:av?{revalidate:0,expire:void 0}:n.cacheControl})}return(0,A.sendRenderResult)({req:a,res:b,generateEtags:ad.generateEtags,poweredByHeader:ad.poweredByHeader,result:x.default.fromStatic(q.rscData,t.RSC_CONTENT_TYPE_HEADER),cacheControl:n.cacheControl})}let u=q.html;if(!p||N||ap)return(0,A.sendRenderResult)({req:a,res:b,generateEtags:ad.generateEtags,poweredByHeader:ad.poweredByHeader,result:u,cacheControl:n.cacheControl});if(as)return u.push(new ReadableStream({start(a){a.enqueue(z.ENCODED_TAGS.CLOSED.BODY_AND_HTML),a.close()}})),(0,A.sendRenderResult)({req:a,res:b,generateEtags:ad.generateEtags,poweredByHeader:ad.poweredByHeader,result:u,cacheControl:{revalidate:0,expire:void 0}});let w=new TransformStream;return u.push(w.readable),m({span:c,postponed:q.postponed,fallbackRouteParams:null}).then(async a=>{var b,c;if(!a)throw Object.defineProperty(Error("Invariant: expected a result to be returned"),"__NEXT_ERROR_CODE",{value:"E463",enumerable:!1,configurable:!0});if((null==(b=a.value)?void 0:b.kind)!==v.CachedRouteKind.APP_PAGE)throw Object.defineProperty(Error(`Invariant: expected a page response, got ${null==(c=a.value)?void 0:c.kind}`),"__NEXT_ERROR_CODE",{value:"E305",enumerable:!1,configurable:!0});await a.value.html.pipeTo(w.writable)}).catch(a=>{w.writable.abort(a).catch(a=>{console.error("couldn't abort transformer",a)})}),(0,A.sendRenderResult)({req:a,res:b,generateEtags:ad.generateEtags,poweredByHeader:ad.poweredByHeader,result:u,cacheControl:{revalidate:0,expire:void 0}})};if(!aF)return await aE.withPropagatedContext(a.headers,()=>aE.trace(i.BaseServerSpan.handleRequest,{spanName:`${aD} ${a.url}`,kind:g.SpanKind.SERVER,attributes:{"http.method":aD,"http.target":a.url}},p));await p(aF)}catch(b){throw b instanceof B.NoFallbackError||await K.onRequestError(a,b,{routerKind:"App Router",routePath:G,routeType:"render",revalidateReason:(0,f.c)({isRevalidate:ak,isOnDemandRevalidate:ah})},ac),b}}},59405:(a,b,c)=>{"use strict";c.d(b,{A:()=>d});let d=(0,c(23339).A)("ChevronRight",[["path",{d:"m9 18 6-6-6-6",key:"mthhwq"}]])},59798:(a,b,c)=>{"use strict";c.r(b),c.d(b,{default:()=>d});let d=(0,c(97954).registerClientReference)(function(){throw Error("Attempted to call the default export of \"/Users/duskfall/Ktiseos-Nyx-Trainer/frontend/app/docs/page.tsx\" from the server, but it's on the client. It's not possible to invoke a client function from the server, it can only be rendered as a Component or passed to props of a Client Component.")},"/Users/duskfall/Ktiseos-Nyx-Trainer/frontend/app/docs/page.tsx","default")},63033:a=>{"use strict";a.exports=require("next/dist/server/app-render/work-unit-async-storage.external.js")},74097:(a,b,c)=>{"use strict";c.d(b,{A:()=>d});let d=(0,c(23339).A)("BookOpen",[["path",{d:"M2 3h6a4 4 0 0 1 4 4v14a3 3 0 0 0-3-3H2z",key:"vv98re"}],["path",{d:"M22 3h-6a4 4 0 0 0-4 4v14a3 3 0 0 1 3-3h7z",key:"1cyq3y"}]])},78583:(a,b,c)=>{"use strict";c.d(b,{A:()=>d});let d=(0,c(23339).A)("ExternalLink",[["path",{d:"M15 3h6v6",key:"1q9fwt"}],["path",{d:"M10 14 21 3",key:"gplh6r"}],["path",{d:"M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6",key:"a6xqqp"}]])},86439:a=>{"use strict";a.exports=require("next/dist/shared/lib/no-fallback-error.external")},90166:(a,b,c)=>{"use strict";c.d(b,{A:()=>h});var d=c(21124),e=c(3991),f=c.n(e),g=c(59405);function h({items:a}){return(0,d.jsx)("nav",{className:"flex items-center space-x-2 text-sm mb-6",children:a.map((b,c)=>{let e=c===a.length-1;return(0,d.jsxs)("div",{className:"flex items-center",children:[c>0&&(0,d.jsx)(g.A,{className:"w-4 h-4 mx-2 text-gray-500"}),b.href&&!e?(0,d.jsxs)(f(),{href:b.href,className:"flex items-center gap-2 text-purple-400 hover:text-purple-300 transition-colors",children:[b.icon,b.label]}):(0,d.jsxs)("span",{className:`flex items-center gap-2 ${e?"text-gray-300 font-semibold":"text-gray-500"}`,children:[b.icon,b.label]})]},c)})})}},91769:(a,b,c)=>{Promise.resolve().then(c.bind(c,59798))}};var b=require("../../webpack-runtime.js");b.C(a);var c=b.X(0,[413,249,37],()=>b(b.s=42416));module.exports=c})();