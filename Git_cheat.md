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


## git rebase -i でコミットをまとめる

1. commitログを確認

`git log --oneline`

	1. コマンド確認

	commit_id_3 commitメッセージ < HEAD
	commit_id_2 commitメッセージ
	commit_id_1 commitメッセージ

これらをまとめたい

2. commitをまとめる

`git rebase -i HEAD~3`

次の画面が表示される

	pick commit_id_1 commitメッセージ
	pick commit_id_2 commitメッセージ
	pick commit_id_3 commitメッセージ

3. fixupにする

コマンド`pick`を`fixup`にする

	pick commit_id_1 commitメッセージ
	fixup commit_id_2 commitメッセージ
	fixup commit_id_3 commitメッセージ

4. viを終了

5. リモートに反映する

`git push -f origin feature`


## git rebase -i で離れたところにあるコミットをまとめる

1. commitログを確認

`git log --oneline`

例えばこんな感じ

	9dc13c0 (HEAD -> master) fix ModuleB
	2bf2d88 fix ModuleA
	249a171 add ModuleB
	5a43974 add ModuleA
	65ef5c7 first commit

5a43974と2bf2d88、249a171と9dc13c0をまとめたい

2. rebaseする

`git rebase -i HEAD~4`

こんな感じのvimが開く

	pick 5a43974 add ModuleA
	pick 249a171 add ModuleB
	pick 2bf2d88 fix ModuleA
	pick 9dc13c0 fix ModuleB

3. 順序を入れ替えてfixupする

	pick 5a43974 add ModuleA
	fixup 2bf2d88 fix ModuleA
	pick 249a171 add ModuleB
	fixup 9dc13c0 fix ModuleB

4. viを終了

5. リモートに反映

`git push -f origin feature`


## コミットメッセージの変更

1. rebaseする

	pick 5a43974 add ModuleA
	pick 249a171 add ModuleB
	pick 2bf2d88 fix ModuleA
	pick 9dc13c0 fix ModuleB

9dc13c0のコミットメッセージを"add ModuleC"に変えたい

2. `pick`を`edit`に変える

	pick 5a43974 add ModuleA
	pick 249a171 add ModuleB
	pick 2bf2d88 fix ModuleA
	edit 9dc13c0 fix ModuleB

3. エディタを終了

こんな感じの画面になる

	You can amend the commit now, with
	git commit --amend 
	Once you are satisfied with your changes, run
	git rebase --continue

 4. commmitする

`git commit --amend`

するとこんな感じになる

	fix ModuleB
	Please enter the commit message for your changes. Lines starting
	with '#' will be ignored, and an empty message aborts the commit.

5. メッセージを修正

	add ModuleC
	Please enter the commit message for your changes. Lines starting
	with '#' will be ignored, and an empty message aborts the commit.

6. エディタを終了、`git rebase --continue`する

7. リモートに反映

`git push -f origin feature`


## branchの削除

### ローカルブランチ

- HEADにマージしたブランチの削除

	`git branch --delete foo`

- マージしたかどうかを問わずに削除

	`git branch -D foo`

### リモートブランチ

- リモートブランチ`foo`を削除

	`git push --delete origin foo`
