# python_article_tag_classify


# 目的
 - 業務で担当しているサイトの記事自動を分類することで、記事テーマごとのアクセスログ解析やモニタリングを行いたい
 
# 仮説
 - 記事テーマごとに、ユーザーのアクセスログのパターンに違いがあるはずなので、分析していきたい。ということで、まずは分類する必要がある。

# やったこと
 - 記事のテキストを全てneolog-dで分かち書き（名詞のみ）
 - TD-IDF
 - 高頻度語を除く
 - トピック分類とクラスター分類を行う
 - 今回はモニタリングのため、1記事1タグで分割したかったため、クラスター分類を採用
 - クラスターごとに頻度が高い単語5個を選定し、簡易タグとして設定

# 結果
 - 納得感のあるタグ分類ができたので、記事のタグ付をこのロジックに変更してモニタリングや詳細な分析を続ける
 
# 課題
 - トピックモデルの最適な数は？
    - モデル評価も行ったが、数値を変えても精度が低い結果になってしまう
    - クラスターに比べて納得感が持ちづらい
 - クラスターの最適な数は？
 　  - いくつか回した結果、人として納得感がある数で設定したが、客観的な根拠は得られないのか？
 - タグの名前の精度を上げられないか？
 　  - クラスターごとに高頻度単語上位5個を見れば推測はできるが、タグとしてユーザーに見せられない。上位1位の単語をタグとするとニュアンスが伝わらない。
 - どのようにしたら自動でタグがつけられるのか？
    - pixivの記事を参考に、BM25と一般辞書をTD-IDFに追加する方法を試してみたい。 
