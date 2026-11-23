import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import { mkdtemp, writeFile, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import { localClassifier, openAIClassification, claudeClassification } from "./provider-tools.mjs";

const exe=path.resolve(process.argv[2]);
const dir=await mkdtemp(path.join(tmpdir(),"bt-provider-test-"));
try {
  const file=path.join(dir,"model.bt");
  await writeFile(file,execFileSync(exe,["probe-init","1536","2"]));
  const vector=Array.from({length:1536},(_,i)=>Math.sin(i));
  const classifier=localClassifier(exe,file);
  const args={features:vector};
  const expected={class_index:0,probabilities:[0.5,0.5]};
  assert.deepEqual(await classifier.call("classify_features",args),expected);
  await assert.rejects(()=>classifier.call("classify_features",{features:[NaN]}));
  await assert.rejects(()=>classifier.call("unrelated",args));
  let openCalls=0, claudeCalls=0;
  const open={responses:{create:async(req)=>{
    if (++openCalls===1) {
      assert.equal(req.tools[0].strict,true);
      return {output:[{type:"function_call",name:"classify_features",call_id:"fixture-o",arguments:JSON.stringify(args)}]};
    }
    assert.equal(req.input.at(-1).call_id,"fixture-o");
    assert.deepEqual(JSON.parse(req.input.at(-1).output),expected);
    return {output_text:"done"};
  }}};
  const claude={messages:{create:async(req)=>{
    if (++claudeCalls===1) {
      assert.equal(req.tools[0].input_schema.required[0],"features");
      return {content:[{type:"tool_use",id:"fixture-a",name:"classify_features",input:args}]};
    }
    assert.equal(req.messages.at(-1).content[0].tool_use_id,"fixture-a");
    assert.deepEqual(JSON.parse(req.messages.at(-1).content[0].content),expected);
    return {content:[{type:"text",text:"done"}]};
  }}};
  assert.equal((await openAIClassification(open,"fixture","classify the features",classifier)).output_text,"done");
  assert.equal((await claudeClassification(claude,"fixture","classify the features",classifier)).content[0].text,"done");
  assert.equal(openCalls,2); assert.equal(claudeCalls,2);
  console.log("Verified both offline provider round trips against the real CLI, including a 1,536-feature vector.");
} finally { await rm(dir,{recursive:true,force:true}); }
