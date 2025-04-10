#!/usr/local/bin/node

var Captcha = module.exports = {

    charArrayFromSource: function (char) {
        var charlist = this.charlist;
        var pbmarr = charlist[char].split(' ');
        var width = +pbmarr[1];
        var height = +pbmarr[2];
        pbmarr.shift();
        pbmarr.shift();
        pbmarr.shift();

        var retArr = [];
        var index = 0;
        retArr[index] = [];

        for (var i = 0, l = pbmarr.length; i < l; i++) {
            retArr[index].push(+pbmarr[i]);
            if (i % width === width - 1 && i < l - 1) {
                index++;
                retArr[index] = [];
            }
        }
        return retArr;
    },

    concatChars: function (chars) {
        var char = [];
        // height
        for (var i = 0, l = chars[0].length; i < l; i++) {
            var concatArr = [];
            // str.length
            for (var j = 0, l2 = chars.length; j < l2; j++) {
                concatArr = concatArr.concat(chars[j][i]);
            }
            char[i] = concatArr;
        }
        return char;
    },

    charArr2str: function (chararr) {
        var width = chararr.length;
        var height = chararr[0].length;
        var outchar = '';

        for (var i = 0; i < width; i++) {
            for (var j = 0; j < height; j++) {
                if (chararr[i][j] === 1) {
                    outchar += '#';
                } else {
                    outchar += ' ';
                }
            }
            outchar += '\n';
        }
        return outchar;
    },

    word2str: function (word) {
        var charsArr = [];
        for (var i = 0, l = word.length; i < l; i++) {
            var charArr = this.charArrayFromSource(word[i]);
            charsArr.push(charArr);
        }
        var oneArr = this.concatChars(charsArr);
        var str = this.charArr2str(oneArr);
        return str;
    },

    newBlankChar: function (width, height) {
        return Array.apply(null, new Array(height)).map(function () {
            return Array.apply(null, new Array(width)).map(Number.prototype.valueOf, 0);
        }, null);
    },

    transformCharArr: function (chararr, transforms) {
        var height = chararr.length;
        var width = chararr[0].length;
        var outchar = '';

        // generate 0-filled array
        var newChar = this.newBlankChar(width, height);

        var newHeight = transforms.newHeight;
        var newWidth = transforms.newWidth;

        for (var i = 0; i < height; i++) {
            for (var j = 0; j < width; j++) {
                var y = i / height;
                var x = j / width;
                // [0,1] x [0,1] -> [0,1]
                var _y = newHeight(x, y) * height;
                // [0,1] x [0,1] -> [0,1]
                var _x = newWidth(x, y) * width;
                if (0 <= _y && _y < height && 0 <= _x && _x < width) {
                    newChar[i][j] = chararr[_y | 0][_x | 0];
                }
            }
        }
        return newChar;
    },

    addFilter2CharArr: function (charArr, transforms) {
        var height = charArr.length;
        var width = charArr[0].length;

        var areaCond = transforms.areaCond;

        for (var i = 0; i < height; i++) {
            for (var j = 0; j < width; j++) {
                var y = i / height;
                var x = j / width;
                // [0,1] x [0,1] -> Boolean
                if (areaCond(x, y)) {
                    charArr[i][j] = 1;
                }
            }
        }
        return charArr;
    },

    word2Transformedstr: function (word) {
        var charsArr = [];
        for (var i = 0, l = word.length; i < l; i++) {
            var charArr = this.charArrayFromSource(word[i]);

            // just for padding top and bottom
            var padding = this.newBlankChar(charArr[0].length, charArr.length / 6 | 0);
            charArr = padding.concat(charArr);
            charArr = charArr.concat(padding);

            charsArr.push(charArr);
        }

        // just for padding left and right
        var padding = this.newBlankChar(charsArr[0][0].length, charsArr[0].length);
        charsArr.unshift(padding);
        charsArr.push(padding);

        var r = Math.random();
        var r2 = Math.random();
        var r3 = Math.random();
        var r4 = Math.random();
        var s = Math.pow(-1, r * 2 | 0);

        var oneArr = this.concatChars(charsArr);
        var oneArr = this.transformCharArr(oneArr, {
            // originally y
            newHeight: function (x, y) {
                return (1 - s * r3 / 10) * (y - s * Math.cos(x * Math.PI * 2 * r2) / (15 - r * y)) + s * r4 / 10;
            },
            // originally x
            newWidth: function (x, y) {
                return (1 + s * r4 / 10) * (x + s * Math.sin(y * Math.PI * 2 * r) / (15 + r2 * x)) - s * r3 / 10;
            }
        });

        var oneArr = this.addFilter2CharArr(oneArr, {
            areaCond: function (x, y) {
                var _x = x - 0.5;
                var _y = y - 0.5;
                var p = Math.pow;
                var cos = Math.cos;
                var expr = cos(4 * x) * (s / r2 * p(_x, 3) - s * p(_x, 2) / r3 + s / r4 * p(_x, 1));
                if (expr < _y && _y < expr + 0.04) {
                    return true;
                } else {
                    return false;
                }
            }
        });

        var str = this.charArr2str(oneArr);
        return str;
    },

    generateRandomText: function (len) {
        var text = "";
        var p = "ABCDEFGHIJKLMNOPQRSTUVWXYZ3456789";
        for (var i = 0; i < len; i++) {
            text += p.charAt(Math.random() * p.length | 0);
        }
        return text;
    },

    charlist: {
        'a': "P1 14 30 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 0 0 0 0 1 1 1 0 0 0 0 1 1 1 1 0 0 1 1 1 0 0 0 0 0 0 1 1 1 0 0 0 1 1 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 1 1 1 1 0 0 0 0 0 1 1 1 1 1 1 1 1 1 0 0 0 1 1 1 1 1 1 1 1 1 1 1 0 0 1 1 1 1 1 0 0 0 0 1 1 1 0 0 1 1 1 0 0 0 0 0 0 1 1 1 0 0 1 1 1 0 0 0 0 0 0 1 1 1 0 0 1 1 1 0 0 0 0 0 1 1 1 1 0 0 1 1 1 1 0 0 1 1 1 1 1 1 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 1 1 1 1 1 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0",
    },

    benchmark: function (sec) {
        var limit = Date.now() + 1000 * sec;
        var i = 0;
        while (Date.now() < limit) {
            this.word2Transformedstr(this.generateRandomText(5)) && i++;
        }
        console.log(i);
    }
};
