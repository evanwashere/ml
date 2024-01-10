import mlx.nn as nn
import mlx.core as mx
import mlx.optimizers as mo

class VectorClassifier(nn.Module):
  def __init__(self, i: int, o: int, hidden: int = 100, dropout = 0.2):
    super().__init__()
    self.d = nn.Dropout(dropout)
    self.l1 = nn.Linear(i, hidden)
    self.l2 = nn.Linear(hidden, o)

  def __call__(self, x):
    x = self.l1(x)
    x = nn.relu(x)

    if self.training:
      x = self.d(x)

    x = self.l2(x)
    x = mx.softmax(x)

    return x

  def fit(self, samples, epochs = 1000, learning_rate = 0.001):
    self.train(True)
    mx.eval(self.parameters())
    optimizer = mo.Adam(learning_rate=learning_rate)

    for e in range(epochs):
      l = []; a = []

      def step(net, i, o):
        x = net(i)
        loss = mx.mean(nn.losses.cross_entropy(x, o))
        accuracy = mx.mean(mx.argmax(x, axis=1) == o)

        return loss, accuracy

      step_fn = nn.value_and_grad(self, step)

      for c, batch in enumerate(samples):
        o = mx.array(batch["label"])
        i = mx.array(batch["vector"])
        (loss, accuracy), grads = step_fn(self, i, o)

        l.append(loss.item())
        a.append(accuracy.item())
        optimizer.update(self, grads)
        mx.eval(self.parameters(), optimizer.state)
        if 0 == (c % 10): print(f"epoch: {e} ({c}), loss: {loss.item():.3f}, accuracy: {accuracy.item():.3f}")

      loss = mx.mean(mx.array(l))
      accuracy = mx.mean(mx.array(a))

      mx.eval([loss, accuracy])
      yield e, loss.item(), accuracy.item()

      samples.reset()

    self.train(False)

# import json
# import mlx.data as dx

# n = VectorClassifier(768, 2, hidden=100)

# def samples(batch = 32):
#   # should be using duckdb and parquet files instead
#   stream = dx.stream_line_reader("./vectors.json", "json")
#   stream = stream.sample_transform(lambda x: transform(bytes(x["json"].tolist())))

#   def transform(buf):
#     x = json.loads(buf)
#     return dict(vector=x["vector"], label=0 if "true" == x["label"] else 1)

#   return stream.shuffle(32768).batch(batch)

# for e, loss, accuracy in n.fit(samples(32), epochs=3):
#   print(f"epoch: {e}, loss: {loss:.3f}, accuracy: {accuracy:.3f}")

# n.save_weights("./model")