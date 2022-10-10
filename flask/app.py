from flask import Flask, url_for, jsonify, redirect, request, render_template

app = Flask(__name__)
app.debug = True
import config
app.config.from_object(config)
books = [
    {"id":1,"name":"三国演义"},
    {"id":2,"name":"水浒传"},
    {"id":3,"name":"红楼梦"},
    {"id":4,"name":"西游记"},
]

@app.route('/index')
def index():
    print("2222222222")
    return "这是首页"


@app.route('/book/<int:book_id>',methods=['GET'])
def book_detail(book_id):
    for book in books:
        if book_id == book['id']:
            print(book_id)
            return book
    return  f"id为{book_id}的图书没有找到"
@app.route("/profile")
def profile():
    user_id = request.args.get("id")
    if user_id:
        return "用户个人中心"
    else:
        return redirect(url_for("index"))
@app.route('/book/list')
def bool_list():
    for book in books:
        book['url'] = url_for("book_detail",book_id=book['id'])
        print(book['url'])
        print(books)
        return jsonify(books)
@app.route('/control')
def control():
    context = {
        'age': 18,
        'books': ['红楼梦', '三国演绎','水浒传','西游记'],
        "person":{"name":"华南辉辉酱",'age':1}
    }
    return render_template("control.html",**context)
@app.route('/about')
def about():
    content = {
        "username":"周杰伦",
        'books':['红楼梦','三国演绎']
    }
    return render_template("about.html",**content)
@app.route('/showimg')
def show_img():
    url = request.url
    hosturl = request.host_url
    fullPath = request.full_path
    username = request.args['username']
    password = request.args['password']
    print("url",url)
    print("hosturl",hosturl)
    print("fullPath",fullPath)
    print("username",username)
    print("password",password)

    return "1111"
if __name__ == '__main__':
    app.run(host='127.0.0.1',port=5000)
