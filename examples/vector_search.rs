use agent::{
  knowledge_base::{chunk::fixed_length_chunking, search::vector_search},
  telemetry,
  tools::web_search::{WebSearchArgs, execute::search},
};
use tiktoken_rs::cl100k_base;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
  telemetry::init()?;

  let web_search_args = WebSearchArgs {
    query: "2026美加墨世界杯最佳射手".to_string(),
    max_results: Some(10),
  };

  // 真正发起搜索，拿到一批网页搜索结果
  let output = search(&web_search_args).await?;

  // 把所有结果的标题 + 内容拼成一整段长文本，模拟「不做任何处理，直接塞给大模型」的情况
  let full_text = output
    .results
    .iter()
    .map(|r| format!("Title: {}\n{}", r.title, r.content))
    .collect::<Vec<_>>()
    .join("\n\n");

  // 大模型不是按字看文字的，而是先切成一个个「token」
  // （token 大致等于"半个词"），API 收费、上下文长度都是按 token 算的。
  //
  // cl100k_base 是 GPT-4/3.5 那一代模型的切法，这里用它来本地估算
  // token 数，不用真的调用 API。
  //
  // 注意：我们实际用的 gpt-4o-mini 是更新的切法（o200k_base），
  // 数字会有点误差，但用来看"压缩前后省了多少"已经够了。
  let enc = cl100k_base()?;

  // 第 2 步：看看这一整段文本如果直接喂给大模型，要花多少 token
  let total_tokens = enc.encode_with_special_tokens(&full_text).len();

  println!("Total characters: {}", full_text.len());
  println!("Total tokens: {}", total_tokens);

  // 第 3 步：把每条搜索结果切成 500 字长、重叠 50 字的小块
  // 重叠是为了避免一句话正好被切断在两个块的边界上，丢失上下文
  let mut all_chunks = Vec::new();
  for result in &output.results {
    let text = format!("Title: {}\n{}", result.title, result.content);
    for chunk in fixed_length_chunking(&text, 500, 50) {
      all_chunks.push(chunk);
    }
  }

  println!("Total chunks: {}", all_chunks.len());

  // 第 4 步：向量检索——注意这里用的是同一个搜索词
  // 但这次不是问整个互联网，而是在刚才切好的这堆小块里，找出真正贴题的那几段
  // vector_search 内部会自己把 query 和每个 chunk 都转成向量，再按余弦相似度排序
  let query = "2026美加墨世界杯最佳射手";
  let hits = vector_search(query, &all_chunks, 3).await?;

  println!("\nQuery: '{query}'");
  println!("{}", "=".repeat(60));
  for (i, hit) in hits.iter().enumerate() {
    // 用 chars() 而不是字节切片，避免在中文字符中间截断导致乱码
    let preview: String = hit.text.chars().take(300).collect();
    println!("\n[{}] Similarity: {:.3}", i + 1, hit.similarity);
    println!("{preview}");
  }

  // 第 5 步：只留下最相关的 3 段，拼起来，看看 token 数降到了多少
  let selected_text = hits
    .iter()
    .map(|hit| hit.text.clone())
    .collect::<Vec<_>>()
    .join("\n\n");
  let selected_tokens = enc.encode_with_special_tokens(&selected_text).len();

  println!("\n{}", "=".repeat(60));
  println!("Total tokens: {total_tokens}");
  println!("Selected tokens: {selected_tokens}");
  // 压缩率：从「全部塞进去」到「只留最相关的几段」，省下了多少 token
  println!(
    "Savings rate: {:.1}%",
    (1.0 - selected_tokens as f64 / total_tokens as f64) * 100.0
  );

  Ok(())
}
