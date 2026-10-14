export class ImageClassificationException extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'ImageClassificationException';
  }
}
