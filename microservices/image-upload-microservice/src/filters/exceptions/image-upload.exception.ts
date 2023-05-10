export class ImageUploadException extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'ImageUploadException';
  }
}
