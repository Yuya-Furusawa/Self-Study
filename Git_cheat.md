# Git関連で忘れちゃうことのまとめ

## fork元の更新を取り込む

1. リモートを登録する

`git remote add root_branch https://github.com/(Fork元のユーザ名)/(フォークしたいリポジトリ.git)`

	1. 登録の確認

	`git remote -v`

2. fork元の変更を取ってくる

`git fetch root_branch`

3. 変更をローカルに反映

`git merge root_branch/master`

4. リモートに反映

`git push origin master`


## git rebase

1. rebaseさせたいブランチに移動する

`git checkout feature`

2. rebase先のブランチを指定してrebaseをする

`git rebase master`

3. コンフリクトが起きた場合はコンフリクト部分を修正して再び`git rebase --continue`する

ちなみに`<<<<HEAD`や`======`などは削除する

4. コンフリクトが解消されるまでやり続ける

5. リモートに反映する

`git push -f origin feature`