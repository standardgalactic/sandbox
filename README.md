# NCL_sandbox
Is a collection of NULL Convention Logic (NCL) circuits and libraries written in Verilog to provide the experience of logically determined design in a familiar context. The tools used are Icarus Verilog and gtkwave. The collection begins with the most basic circuits and will progress to more functionality and complexity.

See [this video](http://www.youtube.com/watch?v=zL1K6K1g6NY) and [this website](http://www.karlfant.net/NCLsandbox) to get started.

<p style="font-family: sans-serif;">
    This text will be displayed in a sans-serif font.

 </p>

   <div style="font-style: italic;">
    Everything within this division tag will be italicized, including <strong>bold</strong> and regular text.
    
    The quick brown fox jumps over the lazy dog. 
    </div>

<p>Edit the text below:</p>
<div contenteditable="true" style="border: 1px solid gray; min-height: 50px; padding: 8px;">
    Click here to start typing...
</div>

<h1>Simple Calculator</h1>
    <input type="number" id="value1" placeholder="Enter a number">
    <input type="number" id="value2" placeholder="Enter another number">
    <button onclick="calculateSum()">Calculate Sum</button>
    <p>Result: <span id="result">0</span></p>

    <script>
        function calculateSum() {
            var val1 = document.getElementById('value1').value;
            var val2 = document.getElementById('value2').value;
            var sum = Number(val1) + Number(val2);
            document.getElementById('result').textContent = sum;
        }
    </script>





This site is deployed at https://standardgalactic.github.io/sandbox/
