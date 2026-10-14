export class ImageSavingException extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'ImageSavingException';
  }
}
